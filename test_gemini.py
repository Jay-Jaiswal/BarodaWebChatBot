import os
from dotenv import load_dotenv
from google import genai


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        print("Gemini smoke test skipped: set GEMINI_API_KEY and GEMINI_MODEL.")
        return 0

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini API working",
    )
    print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
