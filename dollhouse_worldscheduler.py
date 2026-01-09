"""
Dollhouse World Scheduler: Global time/event system.
Manages day/night cycles, agent states, and contextual nudges.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional
import math

# ============================================================================
# TIME & SEASONS
# ============================================================================

class TimeOfDay(Enum):
    DAWN = "dawn"          # 5:00-7:00
    MORNING = "morning"    # 7:00-12:00
    AFTERNOON = "afternoon" # 12:00-17:00
    EVENING = "evening"    # 17:00-20:00
    NIGHT = "night"        # 20:00-5:00

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

# ============================================================================
# AGENT STATE
# ============================================================================

@dataclass
class AgentState:
    """Tracked state for each agent"""
    agent_id: int
    energy: float = 100.0          # 0-100, decreases over time, sleep restores
    hunger: float = 50.0           # 0-100, increases over time
    happiness: float = 70.0        # 0-100, affected by events
    creativity: float = 80.0       # 0-100, peaks during morning
    alertness: float = 70.0        # 0-100, low at night
    social_need: float = 40.0      # 0-100, increases over time
    is_sleeping: bool = False
    sleep_duration: int = 0        # frames slept
    last_interaction: int = -1000  # frame of last interaction
    
    def to_context_string(self) -> str:
        """Generate context string for LLM"""
        states = []
        if self.is_sleeping:
            states.append(f"sleeping (for {self.sleep_duration} frames)")
        if self.energy < 30:
            states.append("exhausted")
        if self.hunger > 70:
            states.append("starving")
        if self.happiness > 80:
            states.append("euphoric")
        if self.creativity > 85:
            states.append("inspired")
        
        return " | ".join(states) if states else "neutral"

# ============================================================================
# WORLD TIME
# ============================================================================

class WorldTime:
    """Global time system"""
    
    def __init__(self, start_frame: int = 0, fps: int = 60):
        self.frame = start_frame
        self.fps = fps
        self.clock_cycle = 14400  # Frames per 24-hour cycle (240 seconds @ 60 FPS)
    
    def advance(self, frames: int = 1):
        """Advance world time"""
        self.frame += frames
    
    @property
    def seconds(self) -> float:
        """Seconds since world start"""
        return self.frame / self.fps
    
    @property
    def world_seconds(self) -> float:
        """Seconds in current day cycle (0-14400)"""
        return self.seconds % (self.clock_cycle / self.fps)
    
    @property
    def hours(self) -> float:
        """Hours in current day (0-24)"""
        return (self.world_seconds / (self.clock_cycle / self.fps)) * 24
    
    @property
    def time_of_day(self) -> TimeOfDay:
        """Current time period"""
        h = self.hours
        if 5 <= h < 7:
            return TimeOfDay.DAWN
        elif 7 <= h < 12:
            return TimeOfDay.MORNING
        elif 12 <= h < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= h < 20:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT
    
    @property
    def day_progress(self) -> float:
        """0-1 progress through day"""
        return (self.hours % 24) / 24
    
    @property
    def brightness(self) -> float:
        """0-1 brightness (for lighting)"""
        h = self.hours % 24
        # Peak brightness at noon, low at night
        if 6 <= h < 18:
            return 0.5 + 0.5 * math.cos(math.pi * (h - 12) / 6)
        else:
            return 0.2  # Night baseline
    
    @property
    def season(self) -> Season:
        """Current season (based on day of year)"""
        day_of_year = int(self.seconds // (86400 * self.fps)) % 365
        if day_of_year < 90:
            return Season.SPRING
        elif day_of_year < 180:
            return Season.SUMMER
        elif day_of_year < 270:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def __str__(self) -> str:
        h = int(self.hours)
        m = int((self.hours % 1) * 60)
        return f"{h:02d}:{m:02d} ({self.time_of_day.value}) - {self.season.value}"

# ============================================================================
# EVENTS
# ============================================================================

@dataclass
class WorldEvent:
    """Scheduled world event"""
    name: str
    time_of_day: Optional[TimeOfDay]  # When it occurs, None = any time
    frame_trigger: Optional[int]       # Specific frame, None = time-based
    callback: Callable[[int], None]    # Function to call with agent_id
    description: str = ""
    cooldown: int = 0                  # Frames before can trigger again
    last_triggered: int = -9999

# ============================================================================
# SCHEDULER
# ============================================================================

class WorldScheduler:
    """Master scheduler for world state"""
    
    def __init__(self, fps: int = 60):
        self.world_time = WorldTime(fps=fps)
        self.agent_states: Dict[int, AgentState] = {}
        self.events: List[WorldEvent] = []
        self.fps = fps
        
        self._init_agents(9)
        self._init_events()
    
    def _init_agents(self, num_agents: int):
        """Initialize agent states"""
        for i in range(num_agents):
            self.agent_states[i] = AgentState(agent_id=i)
    
    def _init_events(self):
        """Register world events"""
        self.events = [
            # Sunrise nudge
            WorldEvent(
                name="sunrise",
                time_of_day=TimeOfDay.DAWN,
                frame_trigger=None,
                callback=self._event_sunrise,
                description="New day begins, energy restored",
                cooldown=14400,  # Once per day cycle
            ),
            # Morning creativity peak
            WorldEvent(
                name="morning_inspiration",
                time_of_day=TimeOfDay.MORNING,
                frame_trigger=None,
                callback=self._event_morning_inspiration,
                description="Creative energy peaks",
                cooldown=14400,
            ),
            # Afternoon slump
            WorldEvent(
                name="afternoon_slump",
                time_of_day=TimeOfDay.AFTERNOON,
                frame_trigger=None,
                callback=self._event_afternoon_slump,
                description="Energy dips, hunger rises",
                cooldown=14400,
            ),
            # Evening wind-down
            WorldEvent(
                name="evening_calm",
                time_of_day=TimeOfDay.EVENING,
                frame_trigger=None,
                callback=self._event_evening_calm,
                description="Day winds down, reflection sets in",
                cooldown=14400,
            ),
            # Nighttime sleep trigger
            WorldEvent(
                name="sleep_time",
                time_of_day=TimeOfDay.NIGHT,
                frame_trigger=None,
                callback=self._event_sleep_call,
                description="Time to rest",
                cooldown=14400,
            ),
            # Hunger trigger
            WorldEvent(
                name="hunger_buildup",
                time_of_day=None,
                frame_trigger=None,
                callback=self._event_hunger_buildup,
                description="Hunger accumulates",
                cooldown=3600,  # Every 60 seconds
            ),
        ]
    
    # ============================================================================
    # EVENT CALLBACKS
    # ============================================================================
    
    def _event_sunrise(self, agent_id: int):
        """Dawn: restore energy, brighten mood"""
        state = self.agent_states[agent_id]
        state.energy = min(100, state.energy + 30)
        state.happiness = min(100, state.happiness + 20)
        state.is_sleeping = False
        state.alertness = 60
    
    def _event_morning_inspiration(self, agent_id: int):
        """Morning: peak creativity and alertness"""
        state = self.agent_states[agent_id]
        state.creativity = min(100, state.creativity + 25)
        state.alertness = min(100, state.alertness + 30)
        state.energy = min(100, state.energy + 10)
    
    def _event_afternoon_slump(self, agent_id: int):
        """Afternoon: energy dips"""
        state = self.agent_states[agent_id]
        state.energy = max(0, state.energy - 20)
        state.alertness = max(0, state.alertness - 15)
        state.hunger = min(100, state.hunger + 15)
    
    def _event_evening_calm(self, agent_id: int):
        """Evening: reflective, social"""
        state = self.agent_states[agent_id]
        state.social_need = min(100, state.social_need + 25)
        state.creativity = max(0, state.creativity - 10)
        state.alertness = max(0, state.alertness - 20)
    
    def _event_sleep_call(self, agent_id: int):
        """Night: if energy low, trigger sleep"""
        state = self.agent_states[agent_id]
        if state.energy < 50:
            state.is_sleeping = True
            state.sleep_duration = 0
            state.alertness = 0
    
    def _event_hunger_buildup(self, agent_id: int):
        """Steady hunger increase"""
        state = self.agent_states[agent_id]
        state.hunger = min(100, state.hunger + 5)
        if state.hunger > 80:
            state.happiness = max(0, state.happiness - 5)
    
    # ============================================================================
    # UPDATE LOOP
    # ============================================================================
    
    def update(self):
        """Advance world by one frame"""
        self.world_time.advance(1)
        
        # Update all agents
        for agent_id, state in self.agent_states.items():
            self._update_agent_state(agent_id, state)
        
        # Check and trigger events
        self._process_events()
    
    def _update_agent_state(self, agent_id: int, state: AgentState):
        """Update individual agent state each frame"""
        # Natural decay
        state.energy = max(0, state.energy - 0.01)
        state.hunger = min(100, state.hunger + 0.02)
        state.social_need = min(100, state.social_need + 0.01)
        
        # Handle sleep
        if state.is_sleeping:
            state.sleep_duration += 1
            state.energy = min(100, state.energy + 0.15)  # Restore faster during sleep
            state.hunger = max(0, state.hunger - 0.03)    # Decrease hunger during sleep
            
            # Wake up when energy full or after long sleep
            if state.energy >= 95 or state.sleep_duration > 3600:
                state.is_sleeping = False
                state.sleep_duration = 0
        else:
            # Awake state decay
            if self.world_time.time_of_day == TimeOfDay.NIGHT:
                state.alertness = max(0, state.alertness - 0.03)
            else:
                state.alertness = min(100, state.alertness + 0.02)
            
            # Creativity peaks in morning
            if self.world_time.time_of_day == TimeOfDay.MORNING:
                state.creativity = min(100, state.creativity + 0.05)
            else:
                state.creativity = max(0, state.creativity - 0.02)
        
        # Happiness affected by needs
        if state.hunger > 80 or state.energy < 20:
            state.happiness = max(0, state.happiness - 0.05)
        elif state.energy > 70:
            state.happiness = min(100, state.happiness + 0.02)
    
    def _process_events(self):
        """Check and trigger events"""
        current_time = self.world_time.time_of_day
        current_frame = self.world_time.frame
        
        for event in self.events:
            should_trigger = False
            
            # Time-based trigger
            if event.time_of_day and current_time == event.time_of_day:
                if current_frame - event.last_triggered > event.cooldown:
                    should_trigger = True
            
            # Frame-based trigger
            if event.frame_trigger and current_frame == event.frame_trigger:
                should_trigger = True
            
            # Trigger for all agents
            if should_trigger:
                for agent_id in self.agent_states:
                    event.callback(agent_id)
                event.last_triggered = current_frame
    
    # ============================================================================
    # QUERY INTERFACE
    # ============================================================================
    
    def get_agent_context(self, agent_id: int) -> str:
        """Get context string for LLM prompt"""
        state = self.agent_states[agent_id]
        time_str = str(self.world_time)
        
        context = f"""
WORLD STATE:
- Time: {time_str}
- Brightness: {self.world_time.brightness:.1%}
- Season: {self.world_time.season.value}

YOUR STATE:
- Energy: {state.energy:.0f}/100
- Hunger: {state.hunger:.0f}/100
- Happiness: {state.happiness:.0f}/100
- Creativity: {state.creativity:.0f}/100
- Alertness: {state.alertness:.0f}/100
- Social Need: {state.social_need:.0f}/100
- Status: {state.to_context_string()}
"""
        return context
    
    def get_brightness_filter(self) -> tuple:
        """Get RGB multiplier for visual lighting"""
        b = self.world_time.brightness
        # Warm at dawn/dusk, cool at night
        time = self.world_time.time_of_day
        
        if time == TimeOfDay.DAWN:
            return (b * 1.2, b * 0.9, b * 0.7)  # Warm orange
        elif time == TimeOfDay.MORNING:
            return (b, b * 0.95, b * 0.9)  # Neutral warm
        elif time == TimeOfDay.AFTERNOON:
            return (b, b, b)  # Neutral
        elif time == TimeOfDay.EVENING:
            return (b * 1.1, b * 0.8, b * 0.6)  # Warm orange/red
        else:  # NIGHT
            return (b * 0.6, b * 0.6, b * 0.8)  # Cool blue
    
    def get_time_string(self) -> str:
        """Human-readable time"""
        return str(self.world_time)
    
    def get_agent_state(self, agent_id: int) -> AgentState:
        """Get raw agent state"""
        return self.agent_states.get(agent_id)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    scheduler = WorldScheduler(fps=60)
    
    print("=== WORLD SCHEDULER DEMO ===\n")
    
    # Simulate a day
    for frame in range(14400):  # Full 24-hour cycle
        scheduler.update()
        
        # Print every 1200 frames (4 minutes of real time = 1 hour world time)
        if frame % 1200 == 0:
            print(f"\nFrame {frame}: {scheduler.get_time_string()}")
            for agent_id in range(3):  # Show first 3 agents
                state = scheduler.get_agent_state(agent_id)
                context = scheduler.get_agent_context(agent_id)
                print(f"Agent {agent_id}: {state.to_context_string()}")
    
    print("\n✓ Scheduler demo complete")