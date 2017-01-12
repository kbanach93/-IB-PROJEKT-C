function [ random ] = randomWithRestriction( randMin, randMax, restr )

random = restr;
while(random == restr)
    random = randMin + round(rand * (randMax - randMin));
end

end

