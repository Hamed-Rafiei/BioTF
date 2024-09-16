clc
clear
close all

% Parameters
num_series = 100;  % Number of time series per class
series_length = 150;  % Length of each time series

% Generate training data
X_sin = zeros(num_series, series_length);
X_cos = zeros(num_series, series_length);

for i = 1:num_series
    t = linspace(0, 4*pi, series_length)' + randn(1) * pi;  % Random phase shift
    X_sin(i, :) = sin(t) + randn(series_length, 1) * 0.1;  % Add some noise
    X_cos(i, :) = cos(t) + randn(series_length, 1) * 0.1;  % Add some noise
end


% Combine data and create labels
X = cat(1, X_sin, X_cos);
Y = categorical([zeros(num_series, 1); ones(num_series, 1)]);

% framzie data using BioTF
frame_length = 10;
frame_slide = 3;
for i = 1:size(X,1)
    X_transformed(i, :, :) = BioTF(X(i,:)', frame_length, frame_slide);
end

if frame_length==1
    num_features = 1;  % Number of features
else
    num_features = size(X_transformed,2);  % Number of features
end
% Shuffle the data
rng(1);  % For reproducibility
idx = randperm(2*num_series);
X_transformed = X_transformed(idx, :, :);
Y = Y(idx);

% Split into training and validation sets
train_ratio = 0.8;
train_size = round(2*num_series * train_ratio);
X_train = X_transformed(1:train_size, :, :);
Y_train = Y(1:train_size);
X_val = X_transformed(train_size+1:end, :, :);
Y_val = Y(train_size+1:end);

% make data compatible with network
if frame_length==1
    X_train = num2cell(X_train,2);
    X_val = num2cell(X_val,2);
else
    X_train = squeeze(mat2cell(permute(X_train,[2 3 1]), size(X_train,2), size(X_train,3), ones(1, size(X_train,1))))';
    X_val = squeeze(mat2cell(permute(X_val,[2 3 1]), size(X_val,2), size(X_val,3), ones(1, size(X_val,1))))';
end

% Create and configure LSTM network
numHiddenUnits = 100;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(num_features)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 10);


% Train the network
net = trainNetwork(X_train, Y_train, layers, options);

% Generate test data
num_test = 50;
X_test_sin = zeros(num_test, series_length);
X_test_cos = zeros(num_test, series_length);

for i = 1:num_test
    t = linspace(0, 2*pi, series_length)' + randn(1) * pi;  % Random phase shift
    X_test_sin(i, :) = sin(t) + randn(series_length, 1) * 0.1;  % Add some noise
    X_test_cos(i, :) = cos(t) + randn(series_length, 1) * 0.1;  % Add some noise
end

X_test = cat(1, X_test_sin, X_test_cos);
Y_test = categorical([zeros(num_test, 1); ones(num_test, 1)]);

% framzie data using BioTF

for i = 1:size(X_test,1)
    X_test_transformed(i, :, :) = BioTF(X_test(i,:)', frame_length, frame_slide);
end

% make data compatible with network
if frame_length==1
    X_test_transformed = num2cell(X_test_transformed,2);
else
    X_test_transformed = squeeze(mat2cell(permute(X_test_transformed,[2 3 1]), size(X_test_transformed,2), size(X_test_transformed,3), ones(1, size(X_test_transformed,1))))';
end

% Classify test data
Y_pred = classify(net, X_test_transformed);

% Calculate accuracy
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
disp(['Test Accuracy: ' num2str(accuracy * 100) '%']);


confusionchart(Y_test, Y_pred);
title('Confusion Matrix');





function out = BioTF(data, frame_length, frame_slide)
% Input validation
if nargin < 3
    error('BioTF requires 3 input arguments: data, frame_length, and frame_slide');
end

[Nsample, Nchannel] = size(data);

if frame_length == 1
    out = data;
    return;
end

if frame_length > Nsample
    error('frame_length cannot be larger than the number of samples within a window');
end

if frame_slide <= 0 || frame_slide > frame_length
    error('frame_slide must be positive and not larger than frame_length');
end
% In BioTF function, add error checking:
if mod(Nsample - frame_length, frame_slide) ~= 0
    warning('frame_slide does not evenly divide the series length minus frame_length. This may lead to unexpected behavior.');
end
% Calculate the number of frames
num_frames = floor((Nsample - frame_length) / frame_slide) + 1;

% Pre-allocate output matrix
out = zeros(Nchannel * 4, num_frames); % modify if you want to use fewer frame characteristics

for k = 1:Nchannel
    frame_start = 1;
    for i = 1:num_frames
        frame_end = frame_start + frame_length - 1;

        % Extract current frame
        current_frame = data(frame_start:frame_end, k);

        % Calculate features
        start_value = current_frame(1);
        end_value = current_frame(end);
        max_value = max(current_frame);
        min_value = min(current_frame);

        % Assign features to output
        out_idx = (k-1)*4 + (1:4); %  modify if you want to use fewer frame characteristics
        out(out_idx, i) = [start_value; max_value; min_value; end_value]; %  modify if you want to use fewer frame characteristics

        % Move to next frame
        frame_start = frame_start + frame_slide;
    end
end
end