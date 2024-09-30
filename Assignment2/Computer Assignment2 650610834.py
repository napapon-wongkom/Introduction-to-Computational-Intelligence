import numpy as np
import matplotlib.pyplot as plt

class Robotic_Arm_Fuzzy:
    def __init__(self,distance):
        self.distance = distance
        # Define the universe of discourse for distance and joint speed
        self.distance_set = np.linspace(0, 100, 101)  # Distance input from 0 to 100 units
        self.speed_set = np.linspace(0, 100, 101)     # Joint speed output from 0 to 100 units

        # Fuzzy membership functions for distance
    def distance_zero(self,x):
        return np.maximum(np.minimum((10 - x) / 10, 1), 0)

    def distance_near(self,x):
        return np.maximum(np.minimum((x - 5) / 15, (30 - x) / 15), 0)

    def distance_moderate(self,x):
        membership = np.minimum((x - 20) / 20, (70 - x) / 20)
        return np.maximum(np.minimum(membership, 1), 0)

    def distance_far(self,x):
        return np.maximum(np.minimum((x - 65) / 30, 1), 0)

    # Fuzzy membership functions for joint speed
    def speed_zero(self,x):
        return np.maximum(np.minimum((10 - x) / 10, 1), 0)

    def speed_slow(self,x):
        return np.maximum(np.minimum((x - 10) / 15, (40 - x) / 15), 0)

    def speed_medium(self,x):
        return np.maximum(np.minimum((x - 30) / 20, (70 - x) / 20), 0)

    def speed_fast(self,x):
        return np.maximum(np.minimum((x - 65) / 20, 1), 0)

    # Fuzzy inference rules
    def fuzzy_rule(self,distance_val):
        # Rule 1: If distance is zero, then speed is zero
        rule1 = np.minimum(self.distance_zero(distance_val), self.speed_zero(self.speed_set))
        
        # Rule 2: If distance is near, then speed is slow
        rule2 = np.minimum(self.distance_near(distance_val), self.speed_slow(self.speed_set))
        
        # Rule 3: If distance is moderate, then speed is medium
        rule3 = np.minimum(self.distance_moderate(distance_val), self.speed_medium(self.speed_set))
        
        # Rule 4: If distance is far, then speed is fast
        rule4 = np.minimum(self.distance_far(distance_val), self.speed_fast(self.speed_set))
        
        # Combine rules using maximum (OR operation)
        return np.maximum(rule1, np.maximum(rule2, np.maximum(rule3, rule4)))

    # Defuzzification: Centroid method to get crisp value
    def defuzzify(self,output_fuzzy):
        return np.sum(output_fuzzy * self.speed_set) / np.sum(output_fuzzy)

    # Control system function: Input = distance, Output = joint speed
    def fuzzy_control(self):
        # Apply fuzzy rules
        fuzzy_output = self.fuzzy_rule(self.distance)
        
        # Defuzzify the fuzzy output to get a crisp joint speed
        joint_speed = self.defuzzify(fuzzy_output)
        
        return joint_speed

    # Visualizing the fuzzy sets and output
    def plot_fuzzy_control(self,distance_input):
        plt.figure(figsize=(10, 8))
        
        # Plot membership functions for distance
        plt.subplot(3, 1, 1)
        plt.plot(self.distance_set, self.distance_zero(self.distance_set), label='Zero')
        plt.plot(self.distance_set, self.distance_near(self.distance_set), label='Near')
        plt.plot(self.distance_set, self.distance_moderate(self.distance_set), label='Moderate')
        plt.plot(self.distance_set, self.distance_far(self.distance_set), label='Far')
        plt.axvline(x=distance_input, color='black', linestyle='--', label=f'Input Distance = {distance_input}')
        plt.title('Fuzzy Set: Distance')
        plt.xlabel('Distance')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()

        # Plot membership functions for speed
        plt.subplot(3, 1, 2)
        plt.plot(self.speed_set, self.distance_zero(self.speed_set), label='Zero')
        plt.plot(self.speed_set, self.speed_slow(self.speed_set), label='Slow')
        plt.plot(self.speed_set, self.speed_medium(self.speed_set), label='Medium')
        plt.plot(self.speed_set, self.speed_fast(self.speed_set), label='Fast')
        plt.axvline(x=self.fuzzy_control(), color='black', linestyle='--', label=f'Output Speed = {self.fuzzy_control():.2f}')
        plt.title('Fuzzy Set: Speed')
        plt.xlabel('Speed')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()

        # Plot the result of fuzzy inference
        plt.subplot(3, 1, 3)
        fuzzy_output = self.fuzzy_rule(self.distance)
        plt.plot(self.speed_set, fuzzy_output, label='Fuzzy Output')
        plt.fill_between(self.speed_set, 0, fuzzy_output, alpha=0.3)
        plt.axvline(x=self.fuzzy_control(), color='black', linestyle='--', label=f'Output Speed = {self.fuzzy_control():.2f}')
        plt.title(f'Fuzzy Output for Distance = {distance_input}')
        plt.xlabel('Speed')
        plt.ylabel('Membership Degree')
        plt.grid()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    distance_input = float(input("Enter Distance of robotic end-effector: "))
    UR5 = Robotic_Arm_Fuzzy(distance_input)
    joint_speed = UR5.fuzzy_control()
    print(f'For distance = {distance_input}, the joint speed is: {joint_speed:.2f}')

    # Plot the fuzzy control system for the given distance input
    UR5.plot_fuzzy_control(distance_input)