%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Fall 2016 
%% Final Project - Thomas Cox VGG-19 dogs-vs-cats classifier

%% Part 1: Download and inspect Pre-trained Convolutional Neural Network (CNN)
%% 1.1: Download pre-trained VGG-19 CNN
vgg19;
convnet = vgg19;

%% 1.2: Inspect CNN
convnet.Layers

%% Part 2: Set up image data
%% 2.1: Pre-process and partition dataset
% load new training image data set as an image datastore
clear categories;
images = imageDatastore('./data/PetImagesKaggleTrain', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
tbl = countEachLabel(images) 

% Use the smallest overlap set
% (useful when the two classes have different number of elements)
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
images = splitEachLabel(images, minSetCount, 'randomize');

%format the images
images.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Use 80% of the images for training, 20% for validation, and 20% for testing.
% splitEachLabel splits the images datastore into two new datastores.
[trainingImages,validationImages, testingImages] = splitEachLabel(images,0.6, 0.2,'randomized');

%% 2.2: Setting up VGG
layersTransfer = convnet.Layers(1:end-3);

numClasses = numel(categories(trainingImages.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%set up our training options
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

VGGnetTransfer = trainNetwork(trainingImages,layers,options);

predictedLabels = classify(netTransfer,testingImages);

% display sample classified images
numTestImages = numel(testingImages.Labels);
idx = randperm(numTestImages,16);
figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

% Check accuracy of model on unseen image data set
accuracy = (mean(predictedLabels == testingImages.Labels)) * 100