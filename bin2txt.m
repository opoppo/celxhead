clc;clear;
global X Y bytesPerEvent
X=768;
Y=640;
bytesPerEvent=4;
%************binµ½mat*************
binName='Event.bin';
binDir='C:\Users\Tony.W\Desktop\celx';

binPath=fullfile(binDir,binName);
eventsMatDir='.';
eventsMatName='events.mat';
eventsMatPath=fullfile(eventsMatDir,eventsMatName);

events=getAllEventsAndSaveAsMat(binPath,eventsMatPath);

%************matµ½txt*************
load('events.mat');
% % use the front 500000 rows
% events_t = events_t(1:500000);
% events_x = events_x(1:500000);
% events_y = events_y(1:500000);
% events_adc = events_adc(1:500000);

idxs=[events_t',events_x',events_y',events_adc'];
dlmwrite('events.txt', idxs, 'delimiter', ' ');