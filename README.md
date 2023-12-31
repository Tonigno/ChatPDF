# ChatPDF
Simple AI tool to chat with your PDF files

The app needs to be configured with your account's secret key which is available on the OpenAI [website](https://platform.openai.com/account/api-keys). Either set it as the `OPENAI_API_KEY` environment variable before using the library:

```bash
export OPENAI_API_KEY='sk-...'
```

Or set `openai.api_key` to its value:

```python
import openai
openai.api_key = "sk-..."
```

Note that this is only a prototype, exception handling has not been implemented, and you may need to properly set the token number for calls to the OpenAI API to avoid exceptions.
