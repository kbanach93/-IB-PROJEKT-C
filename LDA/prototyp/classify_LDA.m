Nclass = Nclass(1:12:12000,:);
% SVclass = SVclass(1:500,:);
VEclass = VEclass(1:2:2000,:);
% 
N_train = floor(0.7*size(Nclass,1));
VE_train = floor(0.7*size(VEclass,1));
% SV_train = floor(0.7*size(SVclass,1));

% Training_samples = cat(1,Nclass(1:N_train,1:4), VEclass(1:VE_train,1:4), SVclass(1:SV_train,1:4));
% 
% Test_samples = cat(1,Nclass(N_train+1:end,1:4), VEclass(VE_train+1:end,1:4), SVclass(SV_train:end,1:4));
% 
% group = cat(1, Nclass(1:N_train,7), VEclass(1:VE_train,7), SVclass(1:SV_train,7));

Training_samples = cat(1, VEclass(1:VE_train,1:4), Nclass(1:N_train,1:4));

Test_samples = cat(1, VEclass(VE_train+1:end,1:4), Nclass(N_train+1:end,1:4));

group = cat(1, VEclass(1:VE_train,7), Nclass(1:N_train,7));


[result]  = classify(Test_samples, Training_samples, group, 'linear');

filename = 'results.xlsx';
xlswrite(filename,result);

figure,
F1 = gscatter(Test_samples(:,3),Test_samples(:,4), result, 'rb', 'v^', 10, 'off');
set(F1, 'LineWidth',2);
legend('Klasa N', 'Klasa VEB', 'Location', 'East');
xlabel('Stosunek amplitudy dodatniej do ujemnej'); ylabel('Stosunek pola do obwodu');

figure,
F1 = gscatter(Test_samples(:,1),Test_samples(:,2), result, 'rb', 'v^', 10, 'off');
set(F1, 'LineWidth',2);
legend('Klasa N', 'Klasa VEB', 'Location', 'East');
xlabel('Amplituda za³amka R [mV]'); ylabel('Energia zespo³u QRS [mV^2]');
