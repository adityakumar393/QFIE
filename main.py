# import numpy as np
# import skfuzzy as fuzz
# from QFIE.FuzzyEngines import QuantumFuzzyEngine

# # --- 1. Define Universes and Fuzzy Sets ---
# # Define the ranges for each variable
# env_light = np.linspace(120, 220, 200)
# changing_rate = np.linspace(-10, 10, 200)
# dimmer_control = np.linspace(0, 10, 200)

# # Define the fuzzy membership functions for 'env_light'
# l_dark = fuzz.trapmf(env_light, [120, 120, 130, 150])
# l_medium = fuzz.trapmf(env_light, [130, 150, 190, 210])
# l_light = fuzz.trapmf(env_light, [190, 210, 220, 220])

# # Define fuzzy sets for 'changing_rate'
# r_ns = fuzz.trimf(changing_rate, [-10, -10, 0])
# r_zero = fuzz.trimf(changing_rate, [-10, 0, 10])
# r_ps = fuzz.trimf(changing_rate, [0, 10, 10])

# # Define fuzzy sets for the output 'dimmer_control'
# dm_vs = fuzz.trapmf(dimmer_control, [0, 0, 2, 4])
# dm_s = fuzz.trimf(dimmer_control, [2, 4, 6])
# dm_b = fuzz.trimf(dimmer_control, [4, 6, 8])
# dm_vb = fuzz.trapmf(dimmer_control, [6, 8, 10, 10])

# # --- 2. Define the Fuzzy Rules ---
# rules = [
#     'if env_light is dark and change_rate is neg_small then dimmer_ctrl is very_big',
#     'if env_light is medium and change_rate is pos_small then dimmer_ctrl is small',
#     'if env_light is light and change_rate is zero then dimmer_ctrl is small',
#     'if env_light is light and change_rate is neg_small then dimmer_ctrl is big'
# ]

# # --- 3. Initialize and Configure the Quantum Fuzzy Engine ---
# # We use 'linear' encoding which is easier to understand.
# qfie = QuantumFuzzyEngine(verbose=True, encoding='linear')

# # Add the input variables
# qfie.input_variable(name='env_light', range=env_light)
# qfie.input_variable(name='change_rate', range=changing_rate)

# # Add the output variable
# qfie.output_variable(name='dimmer_ctrl', range=dimmer_control)

# # Add the fuzzy sets to each variable
# qfie.add_input_fuzzysets(var_name='env_light', set_names=['dark', 'medium', 'light'], sets=[l_dark, l_medium, l_light])
# qfie.add_input_fuzzysets(var_name='change_rate', set_names=['neg_small', 'zero', 'pos_small'], sets=[r_ns, r_zero, r_ps])
# qfie.add_output_fuzzysets(var_name='dimmer_ctrl', set_names=['very_small', 'small', 'big', 'very_big'], sets=[dm_vs, dm_s, dm_b, dm_vb])

# # Set the rules
# qfie.set_rules(rules)

# # --- 4. Build the Quantum Circuit ---
# # Provide the crisp input values here. Let's say light is 170 and the rate is 0.
# crisp_inputs = {'env_light': 170, 'change_rate': 0}
# # 'distributed=False' creates a single quantum circuit for the whole system.
# qfie.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)

# # --- 5. Execute and Get the Result ---
# # Run the circuit on the simulator for 1000 shots.
# crisp_output, activation_values = qfie.execute(n_shots=1000, plot_histo=True)

# print(f"\nCrisp Input: {crisp_inputs}")
# print(f"Crisp Output (Dimmer Level): {crisp_output}")














import numpy as np
import skfuzzy as fuzz
from QFIE.FuzzyEngines import QuantumFuzzyEngine

# --- 1. Define Universes and Fuzzy Sets for the Inverted Pendulum ---

# Define the ranges for each variable
# Angle in degrees from vertical
pendulum_angle = np.linspace(-50, 50, 200)
# Angular velocity in degrees per second
angular_velocity = np.linspace(-10, 10, 200)
# Force in Newtons to be applied to the cart
cart_force = np.linspace(-20, 20, 200)

# --- Define Fuzzy Sets for Input: 'pendulum_angle' ---
angle_neg = fuzz.trapmf(pendulum_angle, [-50, -50, -25, 0])
angle_zero = fuzz.trimf(pendulum_angle, [-25, 0, 25])
angle_pos = fuzz.trapmf(pendulum_angle, [0, 25, 50, 50])

# --- Define Fuzzy Sets for Input: 'angular_velocity' ---
vel_neg = fuzz.trapmf(angular_velocity, [-10, -10, -5, 0])
vel_zero = fuzz.trimf(angular_velocity, [-5, 0, 5])
vel_pos = fuzz.trapmf(angular_velocity, [0, 5, 10, 10])

# --- Define Fuzzy Sets for Output: 'cart_force' ---
force_neg_big = fuzz.trapmf(cart_force, [-20, -20, -15, -10])
force_neg_small = fuzz.trimf(cart_force, [-15, -7.5, 0])
force_zero = fuzz.trimf(cart_force, [-7.5, 0, 7.5])
force_pos_small = fuzz.trimf(cart_force, [0, 7.5, 15])
force_pos_big = fuzz.trapmf(cart_force, [10, 15, 20, 20])


# --- 2. Define the Fuzzy Rules for Balancing the Pendulum ---
# These rules decide how much force to apply based on the pendulum's state.
rules = [
    'if pendulum_angle is zero and angular_velocity is zero then cart_force is zero',
    'if pendulum_angle is pos and angular_velocity is zero then cart_force is pos_small',
    'if pendulum_angle is neg and angular_velocity is zero then cart_force is neg_small',
    'if pendulum_angle is zero and angular_velocity is pos then cart_force is pos_big',
    'if pendulum_angle is zero and angular_velocity is neg then cart_force is neg_big',
    'if pendulum_angle is pos and angular_velocity is pos then cart_force is pos_big',
    'if pendulum_angle is neg and angular_velocity is neg then cart_force is neg_big'
]

# --- 3. Initialize and Configure the Quantum Fuzzy Engine ---
qfie = QuantumFuzzyEngine(verbose=True, encoding='linear')

# Add the input variables
qfie.input_variable(name='pendulum_angle', range=pendulum_angle)
qfie.input_variable(name='angular_velocity', range=angular_velocity)

# Add the output variable
qfie.output_variable(name='cart_force', range=cart_force)

# Add the fuzzy sets to each variable
qfie.add_input_fuzzysets(var_name='pendulum_angle', set_names=['neg', 'zero', 'pos'], sets=[angle_neg, angle_zero, angle_pos])
qfie.add_input_fuzzysets(var_name='angular_velocity', set_names=['neg', 'zero', 'pos'], sets=[vel_neg, vel_zero, vel_pos])
qfie.add_output_fuzzysets(var_name='cart_force', set_names=['neg_big', 'neg_small', 'zero', 'pos_small', 'pos_big'], sets=[force_neg_big, force_neg_small, force_zero, force_pos_small, force_pos_big])

# Set the rules
qfie.set_rules(rules)

# --- 4. Build the Quantum Circuit ---
# Provide the crisp input values.
# Let's test a scenario: The pendulum is leaning slightly to the right (15 degrees)
# and is also falling to the right (velocity of 2 deg/s).
crisp_inputs = {'pendulum_angle': 15, 'angular_velocity': 2}

# 'distributed=False' creates a single quantum circuit for the whole system.
qfie.build_inference_qc(crisp_inputs, draw_qc=True, distributed=False)

# --- 5. Execute and Get the Result ---
# Run the circuit on the simulator for 1000 shots.
crisp_output, activation_values = qfie.execute(n_shots=1000, plot_histo=True)

print(f"\nCrisp Input: {crisp_inputs}")
print(f"Crisp Output (Force to Apply): {crisp_output}")