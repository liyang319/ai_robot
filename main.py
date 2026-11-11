import os
import sys

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY 环境变量。")
        print('请先设置：export DEEPSEEK_API_KEY="你的key"')
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def main() -> None:
    client = get_client()
    messages = []

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
            messages.clear()
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_input})

        print("机器人：", end="", flush=True)
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                stream=True,
            )
            reply_parts = []
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    reply_parts.append(content)
                    print(content, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": "".join(reply_parts)})
        except Exception as exc:
            print(f"\n[调用出错] {exc}")
            messages.pop()


if __name__ == "__main__":
    main()
