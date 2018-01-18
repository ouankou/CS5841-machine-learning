% set random state
rand ('state', sum(100*clock));
randn('state', sum(100*clock));

% set command window format
format compact
format short g

% set some other default values
set(0, 'RecursionLimit', 50);
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultAxesBox', 'on');
set(0,'DefaultLineLineWidth',4);
set(0,'DefaultLineMarkerSize',12);
set(0,'DefaultTextFontSize',16);
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultAxesFontWeight','bold');
set(0,'DefaultTextFontWeight','bold');
set(0,'DefaultAxesColor','white');
set(0,'DefaultFigureColor','white');
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultUicontrolFontSize', 8);
recycle('off');


M = dlmread('airfoil_self_noise.dat');
n = 1;

c = cvpartition(100, 'KFold', 3);
c

T = M(1:1400, 1:6);
T

Te = M(1401:1500, 1:5);
Te

[training, testing, noutput] = preprocess(T, n)

results = myregression(T, Te, n);
