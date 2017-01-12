function [ b, alphas ] = svmTrain( data, labels )

%domyœlne wartoœci
C = 1; tol = 0.0001; sig = 0.5; iterLimit = 10000; passLimit = 10;
[n, d] = size(data);
alphas = zeros(n, 1);
b = 0;

iter = 0; passes = 0;

while((passes < passLimit) && (iter < iterLimit))
    alphaChanged = 0;
    for i = 1:n
        Ei = giveMargin(data(i, :), b, alphas, labels, data, sig) - labels(i);
        if(((labels(i) * Ei < -tol) && alphas(i) < C) || ((labels(i) * Ei > tol) && alphas(i) > 0)) 
            j = randomWithRestriction(1, n, i);
            Ej = giveMargin(data(j, :), b, alphas, labels, data, sig) - labels(j);
            
            ai = alphas(i); aj = alphas(j);
            L = 0; H = C;
            
            if(labels(i) == labels(j))
                L = max(0, ai+aj-H);
                H = min(H, ai+aj);
            else
                L = max(0, aj-ai);
                H = min(H, H-ai+aj);
            end
            if(abs(L - H) < 0.0001)
                continue;
            end
            
            eta = 2 * RBFKernel(data(i, :), data(j, :), sig) - RBFKernel(data(i, :), data(i, :), sig) - RBFKernel(data(j, :), data(j, :), sig);
            if(eta >= 0)
                continue;
            end
            
            newaj = aj - (labels(j) * (Ei - Ej) / eta);
            if(newaj > H) 
                newaj = H;
            end
            if(newaj < L) 
                newaj = L;
            end
            
            if(abs(aj - newaj) < 0.0001)
                continue;
            end
            
            newai = ai + labels(i) * labels(j) * (aj - newaj);
            alphas(j) = newaj; alphas(i) = newai;
            
            b1 = b - Ei - labels(i) * (newai - ai) * RBFKernel(data(i, :), data(i, :), sig) - labels(j) * (newaj - aj) * RBFKernel(data(i, :), data(j, :), sig);
            b2 = b - Ej - labels(i) * (newai - ai) * RBFKernel(data(i, :), data(j, :), sig) - labels(j) * (newaj - aj) * RBFKernel(data(j, :), data(j, :), sig);

            b = 0.5 * (b1 + b2);
            if(newai > 0 && newai < C) 
                b = b1;
            end
            if(newaj > 0 && newaj < C)
                b = b2;
            end
            alphaChanged = 1;
        end
    end
    iter = iter + 1;
    if(alphaChanged == 0)
        passes = passes + 1;
    else
        passes = 0;
    end
end

