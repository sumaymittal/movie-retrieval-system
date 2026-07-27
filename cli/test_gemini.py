import os
from dotenv import load_dotenv
from google import genai


def test_api_key():
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key is not None, "could not find api key"
    assert len(api_key) == 39, "incorrectly loaded api key"

    print(f"Using key {api_key[:6]}...")
    return api_key


def test_get_gemini_response(api_key: str):
    client = genai.Client(api_key=api_key)
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    print(response.text)

    response_metadata = response.usage_metadata
    assert response_metadata is not None, "did not receive metadata in response"

    prompt_tokens = response_metadata.prompt_token_count
    response_tokens = response_metadata.candidates_token_count

    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Response Tokens: {response_tokens}")


def main():
    api_key = test_api_key()
    test_get_gemini_response(api_key)


if __name__ == "__main__":
    main()
