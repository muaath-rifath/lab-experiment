% Aim: Investigate optical communication in a water tank with clear and turbid water

% Define parameters (for a typical water tank experiment)

clear_water_attenuation = 0.1;  % attenuation coefficient for clear water (1/m)

turbid_water_attenuation = 0.5; % attenuation coefficient for turbid water (1/m)

max_distance = 10;  % maximum distance to test in meters (this can be adjusted)

light_power = 1;    % initial light power in arbitrary units (light source intensity)

% Define distances for testing (100 data points for finer resolution)

distances = linspace(0, max_distance, 100);  % Distance from 0 to max_distance

% Simulate light attenuation for clear water using exponential decay

clear_water_signal = light_power * exp(-clear_water_attenuation * distances);

% Simulate light attenuation for turbid water using exponential decay

turbid_water_signal = light_power * exp(-turbid_water_attenuation * distances);

% Plot the received signal strengths for both water types

figure;

hold on;

plot(distances, clear_water_signal, 'b', 'LineWidth', 2);  % Blue for clear water

plot(distances, turbid_water_signal, 'r', 'LineWidth', 2);  % Red for turbid water

xlabel('Distance (m)');

ylabel('Received Signal Strength (Arbitrary Units)');

title('Optical Communication in Water Tank');

legend('Clear Water', 'Turbid Water');

grid on;

hold off;

% Calculate the received signal at the maximum distance for each water type

received_signal_clear_water = clear_water_signal(end);  % Signal strength at max distance for clear water

received_signal_turbid_water = turbid_water_signal(end);  % Signal strength at max distance for turbid water

% Display the results in the MATLAB command window

fprintf('Received Signal at max distance (Clear Water): %.4f\n', received_signal_clear_water);

fprintf('Received Signal at max distance (Turbid Water): %.4f\n', received_signal_turbid_water);

% Calculate the signal loss at max distance for both clear and turbid water

signal_loss_clear_water = light_power - received_signal_clear_water;

signal_loss_turbid_water = light_power - received_signal_turbid_water;

% Display the signal loss

fprintf('Signal Loss (Clear Water) at max distance: %.4f\n', signal_loss_clear_water);

fprintf('Signal Loss (Turbid Water) at max distance: %.4f\n', signal_loss_turbid_water);

