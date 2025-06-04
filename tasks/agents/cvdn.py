from .mp3d_agent import MP3DAgent

class CVDNAgent(MP3DAgent):
    name = "cvdn"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == "navigation_cot_and_decision":
            return self.get_navigation_cot_and_decision_prompt(*args, **kwargs)
        elif task == 'navigation_self_reflective':
            return self.get_navigation_self_reflective_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        # Task
        prompt = '### Instruction: Find the described room according the given dialog. Target: {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Understand the dialog in the Instruction and infer the current progress based on the History and dialog. Then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)
        
        return prompt

    def get_navigation_cot_and_decision_prompt(self, instruction, hist_num, cand_num, cls_token):

        # Task
        prompt = '### Instruction: Find the described room according the given dialog. Target: {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Understand the dialog in the Instruction and infer the current progress based on the History and dialog. '
        prompt += 'Decide the action and generate the navigational reasoning.\n'
        prompt += '- Action Decision: {}'.format(cls_token) + '.' + '\n'
        prompt += '- Navigational Reasoning: '

        return prompt

    def get_navigation_self_reflective_prompt(self, instruction, hist_num, cand_num, cot_pair):
        # Task
        prompt = '### Instruction: Find the described room according the given dialog. Target: {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += f"Choose the correct one from the given two outputs.\n"
        prompt += f'Output1:\n{cot_pair[0]}\n'
        prompt += f'Output2:\n{cot_pair[1]}\n'
        prompt += 'Selection:\n'

        return prompt