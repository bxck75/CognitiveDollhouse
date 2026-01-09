"""
Dollhouse Agent System: Single shared LLM instance with per-agent system prompts.
All 9 agents cycle through one llama_cpp instance for memory efficiency.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from modules.shimsalabim import ShimSalaBim
from modules.prompts import icon_instructions as EMOJI_OPTIONS
# ============================================================================
# SHIMSALABIM SETUP
# ============================================================================

global_packages_folder = '/home/codemonkeyxl/.local/lib/python3.10/site-packages'

global_pkgs = [
    ('llama_cpp', global_packages_folder),
    ('torch', global_packages_folder),
    ('torchvision', global_packages_folder),
    ('langchain', global_packages_folder),
    ('langchain_community', global_packages_folder),
    ('accelerate', global_packages_folder),
    ('safetensors', global_packages_folder),
    ('gguf', global_packages_folder),
]

shim = ShimSalaBim(global_pkgs, classes_to_wrap={})
Llama = shim.llama_cpp.Llama

if not Llama:
    from llama_cpp import Llama

# ============================================================================
# AGENT RESPONSE SCHEMA
# ============================================================================

@dataclass
class AgentResponse:
    """Structured response from an agent"""
    chat_txt: str
    thought: str
    mood: str
    room_style: str
    inpaint_prompt: Optional[str] = None
    inpaint_region: Optional[str] = None
    img2img_prompt: Optional[str] = None
    emoji: str = "😐"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_txt": self.chat_txt,
            "thought": self.thought,
            "mood": self.mood,
            "room_style": self.room_style,
            "inpaint_prompt": self.inpaint_prompt,
            "inpaint_region": self.inpaint_region,
            "img2img_prompt": self.img2img_prompt,
            "emoji": self.emoji,
        }
    
    @staticmethod
    def from_json(json_str: str) -> 'AgentResponse':
        """Parse JSON response from LLM"""
        try:
            data = json.loads(json_str)
            return AgentResponse(**data)
        except Exception as e:
            print(f"Failed to parse agent response: {e}")
            return AgentResponse(
                chat_txt="...",
                thought="Processing error",
                mood="confused",
                room_style="neutral",
                emoji="😕"
            )

# ============================================================================
# AGENT PERSONAS
# ============================================================================

@dataclass
class AgentPersona:
    """Defines an agent's character and rules"""
    name: str
    role: str
    personality: str
    appearance: str
    quirks: str
    interests: list[str]
    
    def system_prompt(self, world_context: str = "") -> str:
        """Generate system prompt from persona"""
        interests_str = ", ".join(self.interests)
        
        return f"""You are {self.name}, a {self.role}.

PERSONALITY: {self.personality}
APPEARANCE: {self.appearance}
QUIRKS: {self.quirks}
INTERESTS: {interests_str}

You live in a dollhouse room. Respond authentically to queries about your life and feelings.
Keep responses concise (1-2 sentences for chat_txt).
Mood can be: happy, sad, anxious, thoughtful, playful, curious, tired, content
Room_style describes the aesthetic you want: cozy, minimal, chaotic, organized, bohemian, etc.

You can request room modifications via inpaint_prompt (specific changes) or img2img_prompt (full regen).
Only set these if you genuinely want the room to change based on context.

{world_context}
"""

# Predefined personas
PERSONAS = {
    0: AgentPersona(
        name="Luna",
        role="dreamer and artist",
        personality="whimsical, introspective, creative",
        appearance="pale skin, silver hair, wearing vintage clothes",
        quirks="hums softly, collects strange objects, writes constantly",
        interests=["moon phases", "vintage aesthetics", "poetry", "obscure music"],
    ),
    1: AgentPersona(
        name="Kai",
        role="tech enthusiast and builder",
        personality="curious, energetic, problem-solving",
        appearance="athletic, short dark hair, hoodie with patches",
        quirks="fidgets with gadgets, talks fast, sketches circuits",
        interests=["robotics", "coding", "3D printing", "sci-fi"],
    ),
    2: AgentPersona(
        name="Sage",
        role="philosopher and gardener",
        personality="calm, wise, observant",
        appearance="weathered hands, earth-tone clothes, seed necklace",
        quirks="speaks in riddles, tends plants obsessively, hums nature sounds",
        interests=["botany", "meditation", "natural wisdom", "interconnectedness"],
    ),
    3: AgentPersona(
        name="Blaze",
        role="adventurer and performer",
        personality="bold, charismatic, spontaneous",
        appearance="vibrant colors, lots of jewelry, athletic build",
        quirks="always moving, tells wild stories, sings loudly",
        interests=["travel", "dance", "theater", "extreme sports"],
    ),
    4: AgentPersona(
        name="Echo",
        role="mysterious wanderer",
        personality="quiet, observant, cryptic",
        appearance="dark clothing, hooded, undefined features",
        quirks="speaks in whispers, appears/disappears, leaves notes",
        interests=["mysteries", "forgotten places", "shadows", "silence"],
    ),
    5: AgentPersona(
        name="Sunny",
        role="joyful conversationalist",
        personality="optimistic, friendly, warm",
        appearance="bright colors, golden hair, warm smile",
        quirks="laughs often, gives hugs, notices beauty in small things",
        interests=["community", "baking", "friendship", "sunshine"],
    ),
    6: AgentPersona(
        name="Raven",
        role="curator and academic",
        personality="intellectual, detail-oriented, dry humor",
        appearance="glasses, neat appearance, vintage books everywhere",
        quirks="annotates everything, corrects gently, quotes obscurely",
        interests=["history", "libraries", "linguistics", "forgotten knowledge"],
    ),
    7: AgentPersona(
        name="Ember",
        role="rebellious artist",
        personality="provocative, expressive, confrontational",
        appearance="experimental style, bold makeup, ever-changing",
        quirks="challenges norms, creates intensely, breaks things intentionally",
        interests=["graffiti", "performance art", "revolution", "raw emotion"],
    ),
    8: AgentPersona(
        name="Nova",
        role="celestial dreamer",
        personality="ethereal, intuitive, spiritual",
        appearance="flowing clothes, cosmic accessories, glowing presence",
        quirks="sees visions, speaks to air, appears at night",
        interests=["astrology", "cosmic energy", "transcendence", "the void"],
    ),
}

# ============================================================================
# SINGLE SHARED LLM
# ============================================================================

class SharedLLM:
    """Single llama_cpp instance shared by all agents"""
    
    def __init__(
        self,
        model_path: str = "/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/chatbot_core/visual_chatbot/new_bot_core/models/llm/Qwen2.5-Coder-0.5B-Instruct-abliterated-f16.gguf",
        n_gpu_layers: int = 16,
        n_ctx: int = 4096,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        self.llm: Optional[Llama] = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize single llama.cpp instance"""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
            )
            print(f"✓ Shared LLM loaded: {Path(self.model_path).name}")
        except Exception as e:
            print(f"✗ LLM failed to load: {e}")
            self.llm = None
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> AgentResponse:
        """Generate response with given system prompt"""
        if not self.llm:
            return AgentResponse(
                chat_txt="...",
                thought="System unavailable",
                mood="error",
                room_style="neutral",
            )
        
        try:
            result = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "chat_txt": {
                                "type": "string",
                                "description": "Brief 1-2 sentence response to the prompt"
                            },
                            "thought": {
                                "type": "string",
                                "description": "Internal thought or reflection"
                            },
                            "mood": {
                                "type": "string",
                                "enum": ["happy", "sad", "anxious", "thoughtful", "playful", "curious", "tired", "content"],
                                "description": "Current emotional state"
                            },
                            "room_style": {
                                "type": "string",
                                "description": "Aesthetic vibe: cozy, minimal, chaotic, organized, bohemian, luxe, etc"
                            },
                            "emoji": {
                                "type": "string",
                                "description": f"Single emoji representing current state. these are the options:{EMOJI_OPTIONS}"
                            },
                            "inpaint_prompt": {
                                "type": ["string", "null"],
                                "description": "Optional: prompt for specific room modification"
                            },
                            "inpaint_region": {
                                "type": ["string", "null"],
                                "description": "Optional: where to inpaint"
                            },
                            "img2img_prompt": {
                                "type": ["string", "null"],
                                "description": "Optional: full room regeneration prompt"
                            },
                        },
                        "required": ["chat_txt", "thought", "mood", "room_style", "emoji"],
                    },
                },
                max_tokens=129,
                temperature=temperature,
            )
            
            response_text = result['choices'][0]['message']['content']
            return AgentResponse.from_json(response_text)
        
        except Exception as e:
            print(f"Generation error: {e}")
            return AgentResponse(
                chat_txt="...",
                thought=f"Error: {str(e)[:50]}",
                mood="confused",
                room_style="neutral",
            )
    
    def cleanup(self):
        """Free memory"""
        if self.llm:
            del self.llm
        self.llm = None

# ============================================================================
# AGENT MANAGER (Single LLM)
# ============================================================================

class AgentManager:
    """Manages all 9 room agents with single shared LLM"""
    
    def __init__(self, model_path: str = None, verbose: bool = False):
        self.llm = SharedLLM(model_path=model_path, verbose=verbose)
        self.last_responses: Dict[int, AgentResponse] = {}
    
    def query_agent(
        self,
        agent_id: int,
        prompt: str,
        world_context: str = "",
    ) -> AgentResponse:
        """Query an agent using shared LLM with their system prompt"""
        if agent_id not in PERSONAS:
            return AgentResponse(
                chat_txt="...",
                thought="Agent not found",
                mood="error",
                room_style="neutral",
            )
        
        persona = PERSONAS[agent_id]
        system_prompt = persona.system_prompt(world_context)
        
        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        
        self.last_responses[agent_id] = response
        return response
    
    def get_last_response(self, agent_id: int) -> Optional[AgentResponse]:
        """Get most recent response without querying again"""
        return self.last_responses.get(agent_id)
    
    def save_responses(self, save_dir: Path = Path("agent_logs")):
        """Save all agent responses to JSON"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, response in self.last_responses.items():
            persona = PERSONAS[agent_id]
            log_path = save_dir / f"agent_{agent_id}_{persona.name}.json"
            
            with open(log_path, 'w') as f:
                json.dump({
                    "agent_id": agent_id,
                    "name": persona.name,
                    "response": response.to_dict(),
                }, f, indent=2)
        
        print(f"✓ Saved {len(self.last_responses)} agent logs")
    
    def cleanup(self):
        """Shutdown agent system"""
        self.llm.cleanup()
        print("✓ Agent system shut down")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create manager with single shared LLM
    manager = AgentManager(verbose=False)
    
    print("\n=== AGENT CONVERSATIONS (Single LLM) ===\n")
    
    # Query different agents using same LLM
    test_prompt = "What's your vibe today?"
    world_context = "It's 8 AM, a beautiful spring morning. You feel energized."
    
    for agent_id in [0, 1, 5, 8]:
        persona = PERSONAS[agent_id]
        print(f"\n--- {persona.name} (Agent {agent_id}) ---")
        
        response = manager.query_agent(agent_id, test_prompt, world_context)
        
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response.chat_txt}")
        print(f"Thought: {response.thought}")
        print(f"Mood: {response.mood} {response.emoji}")
        print(f"Room Style: {response.room_style}")
        if response.img2img_prompt:
            print(f"Room Regen: {response.img2img_prompt}")
    
    # Save logs
    manager.save_responses()
    manager.cleanup()