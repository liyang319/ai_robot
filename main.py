import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_TOOL_ROUNDS = 3

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}

CITY_COORDS = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市当前的实时天气。当用户询问天气、气温、下雨、空气质量等情况时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如：北京"},
            },
            "required": ["city"],
        },
    },
}


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY 环境变量。")
        print('请先设置：export DEEPSEEK_API_KEY="你的key"')
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def build_system_prompt(model: str) -> str:
    return (
        "你是一个中文聊天助手，由 DeepSeek 大模型驱动。"
        f"你当前使用的模型名称（API model ID）是「{model}」。"
        "当用户询问你是什么模型时，直接回答这个模型名称，不要含糊。"
        "当用户询问实时天气时，必须调用 get_weather 工具获取真实数据，"
        "不要凭记忆或猜测回答。"
    )


def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for path in ("/etc/ssl/cert.pem",):
        if os.path.exists(path):
            return ssl.create_default_context(cafile=path)
    return ssl.create_default_context()


_SSL_CONTEXT = _build_ssl_context()


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10, context=_SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _geocode(city: str):
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": city, "count": 1, "language": "zh", "format": "json"}
        )
        results = _http_get_json(url).get("results")
        if results:
            item = results[0]
            return item["latitude"], item["longitude"], item.get("name", city)
    except Exception:
        pass
    if city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        return lat, lon, city
    raise ValueError(f"未找到城市：{city}")


def get_weather(city: str) -> dict:
    lat, lon, name = _geocode(city)
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
    )
    current = _http_get_json(url)["current"]
    return {
        "city": name,
        "condition": WEATHER_CODES.get(current["weather_code"], f"天气代码{current['weather_code']}"),
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity_pct": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "observed_at": current["time"],
        "source": "Open-Meteo",
    }


def run_turn(client: OpenAI, messages: list, model: str) -> None:
    printed_any = False
    for _ in range(MAX_TOOL_ROUNDS):
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[WEATHER_TOOL],
            stream=True,
        )
        content_parts = []
        tool_calls = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if not printed_any:
                    print("机器人：", end="", flush=True)
                    printed_any = True
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    while len(tool_calls) <= tc.index:
                        tool_calls.append({"id": "", "name": "", "arguments": ""})
                    slot = tool_calls[tc.index]
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

        content = "".join(content_parts)

        if not tool_calls:
            if printed_any:
                print()
            messages.append({"role": "assistant", "content": content})
            return

        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
                result = get_weather(args.get("city", ""))
            except Exception as exc:
                result = {"error": str(exc)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    if printed_any:
        print()


def main() -> None:
    client = get_client()
    messages = [{"role": "system", "content": build_system_prompt(DEFAULT_MODEL)}]

    print(f"DeepSeek 聊天机器人（模型：{DEFAULT_MODEL}）")
    print("直接输入内容开始对话；/clear 清空上下文；/exit 退出。\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("再见！")
            break
        if user_input == "/clear":
            messages = [{"role": "system", "content": build_system_prompt(DEFAULT_MODEL)}]
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(client, messages, DEFAULT_MODEL)
        except Exception as exc:
            print(f"\n[调用出错] {exc}")
            messages.pop()


if __name__ == "__main__":
    main()
