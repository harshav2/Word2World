from word2world.utils import euclidean_distance

from word2world.solvers import find_characters

class Evaluator:
    def __init__(self, model, total_input_tokens, total_output_tokens):
        self.model = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

    def euclidean_distance(self, map, previous_maps, world_eval_dictionary):

        euclidean_distance_score = 0
        if len(previous_maps) > 0:
            for previous_map in previous_maps:
                euclidean_distance_score += euclidean_distance(map, previous_map)
            euclidean_distance_score /= len(previous_maps)
        else:
            euclidean_distance_score = 0
        world_eval_dictionary["Average_euclidean_distance"] = euclidean_distance_score

        return world_eval_dictionary

    def evaluate_world(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class Generator:
    def __init__(self, model, total_input_tokens, total_output_tokens):
        self.model = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

    def create_story(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_character_info(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_tileset_info(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def map_tiles_to_chars(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_goals(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_important_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_walkable_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_interactive_object_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def world_generation(self):
        raise NotImplementedError("This method should be overridden by subclasses")
   
    def action_generation(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    

    def feedback_checks(self, rounds, world_eval_dict, previous_eval, story_paragraphs, total_objectives, no_of_important_tiles):
        
        good_feedback_prompt = f"Also, the following is a more detailed feedback of how much you improved in the last generation:\n Your last generation improved the following evaluation metrics, and so you are doing great:\n"
        bad_feedback_prompt = f"\nYour last generation did not improve the following evaluation metrics, and so you need to improve it by being more careful about it:\n"
        good_feedback_check = 0
        bad_feedback_check = 0

        if rounds > 0:
            for key, value in world_eval_dict.items():
                print(f"This round eval: {key}, {value}")
                print(f"Previous round eval: {previous_eval[len(previous_eval) - 1][key]}")

                if key == "astar_path" and world_eval_dict[key] == 0:
                    bad_feedback_check +=2

                if int(world_eval_dict[key]) > int(previous_eval[len(previous_eval) - 1][key]):
                    if key == "agent_reward":
                        good_feedback_check += 2
                    else:
                        good_feedback_check += 1
                    good_feedback_prompt += f"- {key}\n"
                    
                else:
                    if key == "agent_reward":
                        bad_feedback_check += 1
                    else:
                        bad_feedback_check += 1
                    bad_feedback_prompt += f"- {key}\n"
                    
         
            if good_feedback_check == 0:
                good_feedback_prompt = ""
            if bad_feedback_check == 0:
                bad_feedback_prompt = ""
            
            if good_feedback_check >= bad_feedback_check:
                no_of_important_tiles += 1
                story_paragraphs[0] += 1
                story_paragraphs[1] += 1
                total_objectives += 1
        # TODO: Clean code!

        return story_paragraphs, total_objectives, no_of_important_tiles, bad_feedback_prompt, good_feedback_prompt

    


