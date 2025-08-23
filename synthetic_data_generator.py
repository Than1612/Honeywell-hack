import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from typing import Tuple, List
import logging

class DCGAN:
    """
    Deep Convolutional Generative Adversarial Network for synthetic video generation
    """
    
    def __init__(self, latent_dim: int = 100, img_size: Tuple[int, int] = (64, 64)):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = 3
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        logging.info("DCGAN initialized successfully")
    
    def _build_generator(self) -> keras.Model:
        """Build the generator model"""
        model = keras.Sequential([
            # Input layer
            layers.Dense(8 * 8 * 256, input_shape=(self.latent_dim,)),
            layers.Reshape((8, 8, 256)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            # Upsampling layers
            layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Conv2DTranspose(32, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            # Output layer
            layers.Conv2D(self.channels, 3, padding='same', activation='tanh')
        ])
        
        return model
    
    def _build_discriminator(self) -> keras.Model:
        """Build the discriminator model"""
        model = keras.Sequential([
            # Input layer
            layers.Conv2D(64, 3, strides=2, padding='same', input_shape=(*self.img_size, self.channels)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Convolutional layers
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Conv2D(256, 3, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Flatten and output
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def _build_gan(self) -> keras.Model:
        """Build the GAN model"""
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = keras.Model(gan_input, gan_output)
        
        return gan
    
    def compile_models(self, learning_rate: float = 0.0002, beta_1: float = 0.5):
        """Compile all models"""
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Compile GAN
        self.gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
            loss='binary_crossentropy'
        )
        
        logging.info("Models compiled successfully")
    
    def train(self, real_images: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the GAN"""
        # Normalize images to [-1, 1]
        real_images = (real_images.astype(np.float32) - 127.5) / 127.5
        
        # Labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            # Select random batch of real images
            idx = np.random.randint(0, real_images.shape[0], batch_size)
            real_batch = real_images[idx]
            
            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self.generator.predict(noise)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}/{epochs} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
    
    def generate_images(self, num_images: int = 1) -> np.ndarray:
        """Generate synthetic images"""
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        generated_images = self.generator.predict(noise)
        
        # Convert from [-1, 1] to [0, 255]
        generated_images = ((generated_images + 1) * 127.5).astype(np.uint8)
        
        return generated_images

class SyntheticDataGenerator:
    """
    Synthetic data generator for surveillance system edge cases
    """
    
    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize GAN
        self.gan = DCGAN()
        self.gan.compile_models()
        
        logging.info("Synthetic data generator initialized")
    
    def generate_anomaly_scenarios(self, num_scenarios: int = 10):
        """Generate synthetic anomaly scenarios"""
        scenarios = [
            'loitering_person',
            'unusual_movement',
            'object_abandonment',
            'crowd_gathering',
            'suspicious_behavior'
        ]
        
        for scenario in scenarios:
            logging.info(f"Generating {scenario} scenarios...")
            self._generate_scenario_frames(scenario, num_scenarios)
    
    def _generate_scenario_frames(self, scenario: str, num_scenarios: int):
        """Generate frames for a specific scenario"""
        scenario_dir = os.path.join(self.output_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        for i in range(num_scenarios):
            # Generate base frame
            base_frame = self.gan.generate_images(1)[0]
            
            # Apply scenario-specific modifications
            modified_frame = self._apply_scenario_modifications(base_frame, scenario)
            
            # Save frame
            filename = f"{scenario}_{i:03d}.jpg"
            filepath = os.path.join(scenario_dir, filename)
            cv2.imwrite(filepath, modified_frame)
            
            logging.info(f"Generated {filename}")
    
    def _apply_scenario_modifications(self, frame: np.ndarray, scenario: str) -> np.ndarray:
        """Apply scenario-specific modifications to generated frames"""
        if scenario == 'loitering_person':
            # Add stationary person-like objects
            frame = self._add_stationary_objects(frame, num_objects=1)
        
        elif scenario == 'unusual_movement':
            # Add motion blur effect
            frame = self._add_motion_blur(frame)
        
        elif scenario == 'object_abandonment':
            # Add isolated objects
            frame = self._add_isolated_objects(frame, num_objects=2)
        
        elif scenario == 'crowd_gathering':
            # Add multiple person-like objects
            frame = self._add_stationary_objects(frame, num_objects=5)
        
        elif scenario == 'suspicious_behavior':
            # Add unusual patterns
            frame = self._add_unusual_patterns(frame)
        
        return frame
    
    def _add_stationary_objects(self, frame: np.ndarray, num_objects: int) -> np.ndarray:
        """Add stationary person-like objects to frame"""
        frame_copy = frame.copy()
        
        for _ in range(num_objects):
            # Generate random position
            x = np.random.randint(50, frame.shape[1] - 50)
            y = np.random.randint(50, frame.shape[0] - 50)
            
            # Draw simple person-like shape
            cv2.rectangle(frame_copy, (x-20, y-40), (x+20, y+40), (255, 255, 255), -1)
            cv2.circle(frame_copy, (x, y-50), 15, (255, 255, 255), -1)
        
        return frame_copy
    
    def _add_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Add motion blur effect to frame"""
        # Simple motion blur using averaging
        kernel = np.ones((1, 5), np.float32) / 5
        blurred = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original
        return cv2.addWeighted(frame, 0.7, blurred, 0.3, 0)
    
    def _add_isolated_objects(self, frame: np.ndarray, num_objects: int) -> np.ndarray:
        """Add isolated objects to frame"""
        frame_copy = frame.copy()
        
        for _ in range(num_objects):
            # Generate random position
            x = np.random.randint(50, frame.shape[1] - 50)
            y = np.random.randint(50, frame.shape[0] - 50)
            
            # Draw simple object
            size = np.random.randint(15, 30)
            cv2.rectangle(frame_copy, (x-size, y-size), (x+size, y+size), (0, 255, 0), -1)
        
        return frame_copy
    
    def _add_unusual_patterns(self, frame: np.ndarray) -> np.ndarray:
        """Add unusual visual patterns to frame"""
        frame_copy = frame.copy()
        
        # Add random geometric shapes
        for _ in range(3):
            x = np.random.randint(50, frame.shape[1] - 50)
            y = np.random.randint(50, frame.shape[0] - 50)
            
            if np.random.random() > 0.5:
                # Circle
                radius = np.random.randint(10, 25)
                cv2.circle(frame_copy, (x, y), radius, (0, 0, 255), -1)
            else:
                # Triangle
                pts = np.array([[x, y-20], [x-15, y+20], [x+15, y+20]], np.int32)
                cv2.fillPoly(frame_copy, [pts], (255, 0, 0))
        
        return frame_copy
    
    def create_video_sequence(self, scenario: str, num_frames: int = 30, fps: int = 10):
        """Create a video sequence from generated frames"""
        scenario_dir = os.path.join(self.output_dir, scenario)
        if not os.path.exists(scenario_dir):
            logging.error(f"Scenario directory {scenario_dir} not found")
            return
        
        # Get all frames for the scenario
        frame_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            logging.error(f"No frames found in {scenario_dir}")
            return
        
        # Create video writer
        output_path = os.path.join(self.output_dir, f"{scenario}_sequence.mp4")
        first_frame = cv2.imread(os.path.join(scenario_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for i in range(min(num_frames, len(frame_files))):
            frame_path = os.path.join(scenario_dir, frame_files[i % len(frame_files)])
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        logging.info(f"Video sequence created: {output_path}")
    
    def save_models(self, models_dir: str = "models"):
        """Save trained GAN models"""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save generator
        generator_path = os.path.join(models_dir, "generator.h5")
        self.gan.generator.save(generator_path)
        
        # Save discriminator
        discriminator_path = os.path.join(models_dir, "discriminator.h5")
        self.gan.discriminator.save(discriminator_path)
        
        logging.info(f"Models saved to {models_dir}")
    
    def load_models(self, models_dir: str = "models"):
        """Load pre-trained GAN models"""
        try:
            generator_path = os.path.join(models_dir, "generator.h5")
            discriminator_path = os.path.join(models_dir, "discriminator.h5")
            
            if os.path.exists(generator_path) and os.path.exists(discriminator_path):
                self.gan.generator = keras.models.load_model(generator_path)
                self.gan.discriminator = keras.models.load_model(discriminator_path)
                self.gan = self.gan._build_gan()
                logging.info("Pre-trained models loaded successfully")
                return True
            else:
                logging.warning("Pre-trained models not found")
                return False
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return False

def main():
    """Main function for synthetic data generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthetic Data Generator for Surveillance System')
    parser.add_argument('--output', '-o', default='synthetic_data',
                       help='Output directory for generated data')
    parser.add_argument('--scenarios', '-s', type=int, default=10,
                       help='Number of scenarios to generate')
    parser.add_argument('--frames', '-f', type=int, default=30,
                       help='Number of frames per video sequence')
    parser.add_argument('--fps', type=int, default=10,
                       help='FPS for video sequences')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(output_dir=args.output)
    
    # Try to load pre-trained models
    if not generator.load_models():
        logging.info("No pre-trained models found. Training new GAN...")
        # Generate dummy training data (in real scenario, use actual surveillance footage)
        dummy_data = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)
        generator.gan.train(dummy_data, epochs=50)
        generator.save_models()
    
    # Generate anomaly scenarios
    generator.generate_anomaly_scenarios(args.scenarios)
    
    # Create video sequences
    scenarios = ['loitering_person', 'unusual_movement', 'object_abandonment', 'crowd_gathering', 'suspicious_behavior']
    for scenario in scenarios:
        generator.create_video_sequence(scenario, args.frames, args.fps)
    
    logging.info("Synthetic data generation completed!")

if __name__ == '__main__':
    main()
