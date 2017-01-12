function [Qs,Ss,RS]=findqrs(Rs,Fs,n)

%R - struktura/macierz wektor probek ann dla ktorych wykryto R
%funkcja zwraca wektory z numeracja probek dla ktorych jest Q i S
%liczba badanych sygnalow, kazdy sygnal to kolejny wiersz


qrtime=0.063;
rstime=0.094;

qr=round(Fs*qrtime);
rs=round(rstime*Fs);

for j=1:n
    
  
    R=Rs{1,j};
    
    if ~isempty(R)
    
    Q=zeros(1,length(R));
    S=Q;

    for i=1:length(R)
        Q(i)=R(i)-qr;
        S(i)=R(i)+rs;
    end
    
    Qs(j,1:length(R))=Q';
    Ss(j,1:length(R))=S';
    RS(j,1:length(R))=R;
    end
end
    
end