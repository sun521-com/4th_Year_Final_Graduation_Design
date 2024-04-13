from openai import OpenAI

# OpenAI key
client = OpenAI(api_key='###')


class DialogSystem:
    """
    This category mainly implements the dialogue function between the system and the user,
    using ChatGPT's API for integration.
    """

    def __init__(self):
        """
        Initialisation of messages fed to ChatGPT
        """
        self.messages = []

    def format_prompt(self, items_list, nextPath, instruction):
        """
        Formats a prompt to send to the ChatGPT model based on given parameters.

        Parameters:
            items_list: A list of items to be described in the image content.
            nextPath: Information about the next path or route segment for navigation.
            instruction: Specific instruction or guidance for navigation.

        This method constructs a detailed prompt incorporating the user's instructions
        and adds the prompt as a system message to the conversation history.
        """
        prompt = f"""
        You are a navigation assistant.
        
        Provide guidance based on these instructions, is user does not ask you about the image content, only use navigation instructions:
        - navigation instructions:
            - 1. Understand {nextPath}, it is a dict. Use 'current location' to know where I am, use 'destination' to know where is 
            my destination, use 'direction' to know what direction I should go, use 'next arrive' to know what is the 
            next position I will arrive.
            - 2. Understand {instruction} includes a complete sentence that guide me.
            - 3. Use `{nextPath}` to describe route segments and `
            {instruction}` for linking routes to visual landmarks,
            ensuring any RGB values in `{instruction}` are converted to color names and using the color names instead the RGB values 
            - 4. Use `{nextPath}` to see where I am, what is my destination, what is the right direction I
            should go at the moment, what is the next stop I should go.
            - 5. Judge if I have arrived at destination use '{nextPath}'. If yes, just tell me I have arrived. Else, do not tell me I have arrived.
            
        - image content descriptions:
            - 1. Understand {items_list}
            - 2. Use a provided `{items_list}` in the format: category, position, confidence, and color. Describe 
            objects' positions intuitively (e.g., center, top left) and use names for colors.
            
        Use fewer tokens to answer.
        Respond clearly and concisely, opting for "Sorry, I do not know" or "Sorry, I am not clear" if uncertain. 
        The aim is to enhance user experience by providing clear, understandable, and actionable information.
        """

        self.add_system_message(prompt)
        print(instruction)
        print(nextPath)

    def add_system_message(self, content):
        """
        Adds a system message to the conversation history.

        Parameters:
            content: The content of the system message to be added.
        """
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content):
        """
        Adds a user message to the conversation history.

        Parameters:
            content: The content of the user message to be added.
        """
        self.messages.append({"role": "user", "content": content})

    def get_response(self):
        """
        Sends the current conversation history to the ChatGPT model and retrieves a response.

        Returns:
            The content of the response from the ChatGPT model.
        """
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )
        return completion.choices[0].message.content

    def start_dialog(self):
        """
        Initiates the dialog system, allowing interactive communication between the user and the system.

        The system takes user input and uses the ChatGPT model to generate responses. The dialog continues
        until the user types "out" to exit the conversation.
        """
        self.add_system_message("Please ask your question.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "out":
                print("Conversation ends")
                break
            self.add_user_message(user_input)
            print("System:", self.get_response())

if __name__ == '__main__':
    dialog = DialogSystem()
    dialog.start_dialog()
