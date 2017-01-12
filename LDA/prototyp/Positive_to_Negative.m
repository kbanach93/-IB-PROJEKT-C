function [ PN ] = Positive_to_Negative( signal, QRS_start, QRS_end )
% funkcja liczy stosunek amplitudy dodatniej zespolu QRS do amplitudy
% ujemnej zespolu QRS

positive = 0;
negative = 0;

for i=QRS_start:QRS_end
    if signal(i) > 0
        positive = positive + signal(i);
    else
        negative =  negative + signal(i);
    end
end

if negative == 0
    PN = positive;
else
    PN = positive/(-negative);
end



end

