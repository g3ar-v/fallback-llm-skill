import openai as ai
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"


class FallbackChatgpt(CommonQuerySkill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        pass

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_ChatGPT(utt)
        # self.log.info(response)
        if response:
            if "?" in response:
                return None
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_ChatGPT(self, message):
        # prompt = self.build_prompt(message.data['utterance'])
        prompt = self.build_prompt(message)

        self.log.info(f'prompt: {prompt}')
        # self.log.info(type(self.init_prompt))

        try:
            completion = self.chatgpt.create(model=model, messages=prompt,
                                             max_tokens=100, temperature=0.2)
            response = completion.choices[0].message["content"]
            if not response or not response.strip("?") or not response.strip("_"):
                return False
            self.log.info(f'response: {response}')
            unwanted_string = "As an AI language model,"
            response = response.replace(unwanted_string, "")
            return response
        except Exception as e:
            self.log.error(f'error in ChatGPT fallback request: {e}')
            return None

    def build_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]

    @property
    def chatgpt(self):
        # key = self.settings.get("key") or api_key
        if not self.key:
            raise ValueError("Openai key not set in settings.json")
        ai.api_key = self.key
        return ai.ChatCompletion

    def stop(self):
        pass


def create_skill():
    return FallbackChatgpt()
