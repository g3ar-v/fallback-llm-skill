from mycroft import FallbackSkill
import openai as ai

model = "gpt-3.5-turbo"

class FallbackChatgpt(FallbackSkill):
    def __init__(self):
        FallbackSkill.__init__(self)
        self._chat = None
        self.key = self.settings.get("key")

    def initialize(self):
        self.register_fallback(self.handle_fallback_ChatGPT, 8)

    def handle_fallback_ChatGPT(self, message):
        prompt = self.build_prompt(message.data['utterance'])
        self.log.info(f'prompt: {prompt}')

        try:
            completion = self.chatgpt.create(model=model,messages=prompt,
                                             max_tokens=200)
            response = completion.choices[0].message["content"]
            if not response or not response.strip("?") or not response.strip("_"):
                return False
            self.log.info(f'response: {response}')
            self.speak(response)
            return True
        except Exception as e:
            self.log.error(f'error in ChatGPT fallback request: {e}')
            return False

    def build_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]


    @property
    def chatgpt(self):
        # key = self.settings.get("key") or api_key
        if not self.key:
            raise ValueError("Openai key not set in settings.json")
        if not self._chat:
            ai.api_key = self.key
            self._chat
        return ai.ChatCompletion


def create_skill():
    return FallbackChatgpt()
    

if __name__ == "__main__":
    from ovos_utils.messagebus import Message

    gpt = FallbackChatgpt()
    msg = Message("intent_failure", {"utterance": "when will the world end?"})
    gpt.handle_fallback_ChatGPT(msg)
