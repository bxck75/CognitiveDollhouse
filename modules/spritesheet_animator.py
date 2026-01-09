import pygame
from enum import Enum

class Direction(Enum):
    LEFT = -1
    RIGHT = 1

class SpritesheetAnimator:
    """Animates a character spritesheet with walking left/right animation."""
    
    def __init__(self, spritesheet_path, frame_width, frame_height, total_frames, fps=10):
        """
        Args:
            spritesheet_path: Path to the spritesheet image
            frame_width: Width of each frame in pixels
            frame_height: Height of each frame in pixels
            total_frames: Total number of frames in the spritesheet
            fps: Frames per second for animation speed
        """
        self.spritesheet = pygame.image.load(spritesheet_path).convert_alpha()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.total_frames = total_frames
        self.fps = fps
        
        # Animation state
        self.current_frame = 0
        self.frame_counter = 0
        self.direction = Direction.RIGHT
        
        # Extract all frames from spritesheet
        self.frames = self._extract_frames()
        
        # Get current image and rect
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect()
    
    def _extract_frames(self):
        """Extract individual frames from the spritesheet."""
        frames = []
        for i in range(self.total_frames):
            # Calculate position in spritesheet (assumes horizontal layout)
            x = i * self.frame_width
            y = 0
            
            # Create a surface for this frame
            frame_surface = pygame.Surface(
                (self.frame_width, self.frame_height),
                pygame.SRCALPHA
            )
            frame_surface.blit(
                self.spritesheet,
                (0, 0),
                pygame.Rect(x, y, self.frame_width, self.frame_height)
            )
            
            frames.append(frame_surface)
        
        return frames
    
    def set_direction(self, direction):
        """Set movement direction (Direction.LEFT or Direction.RIGHT)."""
        self.direction = direction
    
    def update(self, dt):
        """
        Update animation frame.
        
        Args:
            dt: Delta time in seconds
        """
        # Increment frame counter based on FPS
        self.frame_counter += dt * self.fps
        
        if self.frame_counter >= 1:
            self.frame_counter -= 1
            self.current_frame = (self.current_frame + 1) % self.total_frames
            self._update_image()
    
    def _update_image(self):
        """Update the current image based on frame and direction."""
        frame = self.frames[self.current_frame]
        
        # Flip image if walking left
        if self.direction == Direction.LEFT:
            self.image = pygame.transform.flip(frame, True, False)
        else:
            self.image = frame
    
    def draw(self, surface, x, y):
        """
        Draw the current frame at the specified position.
        
        Args:
            surface: Pygame surface to draw on
            x: X coordinate
            y: Y coordinate
        """
        self.rect.x = x
        self.rect.y = y
        surface.blit(self.image, self.rect)


# Example usage:
if __name__ == "__main__":
    pygame.init()
    
    # Screen setup
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 900
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Spritesheet Animation")
    clock = pygame.time.Clock()
    
    # Create animator (adjust path and frame dimensions to match your spritesheet)
    animator = SpritesheetAnimator(
        spritesheet_path="visuals/characters/video_sprites/spritesheet2L.png",  # Change to your spritesheet path
        frame_width=308,  # Adjust to your frame width
        frame_height=454,  # Adjust to your frame height
        total_frames=32,  # Number of frames in the spritesheet
        fps=5  # Animation speed
    )
    
    # Character position
    char_x = SCREEN_WIDTH // 2 
    char_y = SCREEN_HEIGHT // 2
    
    # Main loop
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            animator.set_direction(Direction.LEFT)
            char_x -= 5
        elif keys[pygame.K_RIGHT]:
            animator.set_direction(Direction.RIGHT)
            char_x += 5
        
        # Update animation
        animator.update(dt)
        
        # Draw
        screen.fill((50, 50, 50))
        animator.draw(screen, char_x-72, char_y - 228)
        pygame.display.flip()
    
    pygame.quit()