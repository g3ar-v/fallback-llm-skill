import openai as ai
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"


class FallbackGPT(CommonQuerySkill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        pass

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            if response.endswith("?"):
                return None
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_GPT(self, message):
        prompt = self.build_prompt(message)

        self.log.info(f'prompt: {prompt}')

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
            self.log.error('error in fallback request: {}'.format(e))
            return None

    def build_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]

    @property
    def chatgpt(self):
        if not self.key:
            raise ValueError("Openai key not set in settings.json")
        ai.api_key = self.key
        return ai.ChatCompletion

    def stop(self):
        pass


def create_skill():
    return FallbackGPT()
