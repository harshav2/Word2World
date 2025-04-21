# Word2World

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/c7e5af2e-a948-4eda-9e9c-4c0e0f0f2f46)

This repository contains to code for [Word2World: Generating Stories and Worlds through Large Language Models](https://arxiv.org/abs/2405.06686).

### Abstract:

Large Language Models (LLMs) have proven their worth across a diverse spectrum of disciplines. LLMs have shown great potential in Procedural Content Generation (PCG) as well, but directly generating a level through a pre-trained LLM is still challenging. This work introduces `Word2World`, a system that enables LLMs to procedurally design playable games through stories, without any task-specific fine-tuning. `Word2World` leverages the abilities of LLMs to create diverse content and extract information. Combining these abilities, LLMs can create a story for the game, design narrative, and place tiles in appropriate places to create coherent worlds and playable games. We test `Word2World` with different LLMs and perform a thorough ablation study to validate each step.

### Usage:

Clone the repo:

`https://github.com/umair-nasir14/Word2World.git`

Install the environment and activate it:

```
cd Word2World
type > word2world/.env
conda env create -f environment.yml
conda activate word2world
```

Add your API key to the .env file created in word2world folder:

```
OPENAI_API_KEY="sk..."
```

Run with default configs:

`python main.py`

Or run with specified configs:

```
python main.py \
--model="gpt-4-turbo-2024-04-09" \
--min_story_paragraphs=4 \
--max_story_paragraphs=5 \
--total_objectives=8 \
--rounds=1 \
--experiment_name="Your_World" \
--save_dir="outputs"
```

To play the generated game:

```
python word2world/play_game.py "path_to_game_data\game_data.json"
```

where `game_data.json` is generated when the Word2World loop is finished and is saved to `\outputs\game_data.json`. This can be modified in `configs` or as `--save_dir` arg.

To play an example world:

```
python word2world/play_game.py
```

### To-dos:

- [ ] Change the code to remove the tiles, and animations (Harsha)
- [ ] Change the LLM to a local model
- [ ] Initialise the graph database
- [ ] Add prompting and more for extracting graph details from the story
- [ ] Include graphDB RAG implementation for better context delivery to LLM
- [ ] Develop strict prompts that ensure the LLM will deny a user action, if it is inconsistent with the game world
- [ ] Extract details of user input, and change the knowledge graph accordingly
