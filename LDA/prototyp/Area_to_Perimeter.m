function [ area_perim ] = Area_to_Perimeter( signal, QRS_start, QRS_end )
% funkcja liczy wspolczynnik ksztaltu zespolu QRS bedacy
% stosunkiem pola powierzchni zespolu QRS do jego obwodu

area = 0;
perim = 0;

for i = QRS_start:QRS_end
    area = area + abs(signal(i));
end

for i = QRS_start:(QRS_end - 1)
    perim = perim + abs(signal(i+1) - signal(i));
end

area_perim = 10*(area/perim);
end

