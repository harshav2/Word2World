import os
import json
from dotenv import load_dotenv
#from .configs import Config

class Word2World:
    def __init__(self):
        self.total_input_tokens = []
        self.total_output_tokens = []
        self.worlds_history = {}
        self.previous_story = []
        self.previous_eval = []
        self.prev_agent_reward = []
        self.total_spent = 0

    def run(self, cfg):

        load_dotenv()

        if "gpt" in cfg.model: 
            import openai

        else:
            raise NotImplementedError("Model not implemented yet!")
        
        

        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        if "gpt" in cfg.model: 
            from .generators import OpenAIGenerator
            generator = OpenAIGenerator(self.total_input_tokens, self.total_output_tokens)
        
        
        story, story_prompt = generator.create_story(cfg.story_paragraphs, cfg.total_objectives)

        character_discriptions, character_discriptions_dict, character_prompt, protagonist_name, antagonist_name = generator.extract_character_info(story, story_prompt)
        
        goal_discriptions, goal_prompt = generator.extract_goals(story, story_prompt, character_discriptions, character_prompt) 

        for rounds in range(cfg.rounds):
            print(f"ROUND # {rounds}\n")
            world_map_fixed, world_map_fixed_with_chars, world_eval_dict, \
            story_paragraphs, objectives, total_objectives, good_feedback_check, bad_feedback_check, \
            agent_reward, astar_path= generator.world_generation(rounds,
                                                                                        self.previous_story,
                                                                                        cfg.story_paragraphs,
                                                                                        cfg.total_objectives,
                                                                                        self.previous_eval,
                                                                                        story, 
                                                                                        story_prompt, 
                                                                                        character_discriptions,
                                                                                        character_discriptions_dict, 
                                                                                        character_prompt,  
                                                                                        goal_discriptions, 
                                                                                        goal_prompt, 
                                                                                        cfg.save_dir)


            
            self.previous_story.append(story['choices'][0]['message']['content'])
            self.previous_eval.append(world_eval_dict)
            self.prev_agent_reward.append(agent_reward)

            self.worlds_history[f"round_{rounds}"] = {"story": story['choices'][0]['message']['content'],
                                                        "character_information": character_discriptions['choices'][0]['message']['content'],
                                                        "goals": goal_discriptions['choices'][0]['message']['content'],
                                                        "objectives": objectives,
                                                        "evaluations": world_eval_dict,
                                                        "complexity": {
                                                            "good_feedback_check": good_feedback_check,
                                                            "bad_feedback_check": bad_feedback_check, 
                                                            "story_paragraphs": story_paragraphs,
                                                            "total_objectives": total_objectives
                                                        }}
            
            with open(cfg.save_dir +f"/data_gen_{cfg.experiment_name}.json", 'w') as f:
                json.dump(self.worlds_history, f)

            spent_this_gen = (sum(self.total_input_tokens)/1000)*0.01 + (sum(self.total_output_tokens)/1000)*0.03 
            self.total_spent += spent_this_gen
            print(f"$ spent on this gen = {spent_this_gen}")
            print(f"Total spent = {self.total_spent}")