%% findqrs

n=7;
Fs=360;
RN={ann105N,ann106N,ann114N,{},ann210N,ann223N,ann233N}; %id=1
RVE={ann105VE,ann106VE,ann114VE,ann118VE,ann210VE,ann223VE,ann233VE}; %id=2
RSV={{},{},ann114SV,ann118SV,ann210SV,ann223SV,ann233SV}; %id=3


[QsVE,SsVE,RsVE]=findqrs(RVE,Fs,n); 
[QsN,SsN,RsN]=findqrs(RN,Fs,n);
[QsSV,SsSV,RsSV]=findqrs(RSV,Fs,n);

signals={sig105,sig106,sig114,sig118,sig210,sig223,sig233};

%% N = 1 (id)

N=n;
class=1;
k=1;
for i=1:N %iteracja po kole
    %m=i;
    signal=signals{1,i};
    
    for j=1:length(QsN)
        if QsN(i,j)>0
           
            %i to kolejny sygnal w strukturze i kolejny wiersz w macierzach
            %j to kolejne probki czy kolejne za?amki QRS
            
            Q=QsN(i,j); %indeks Q
            S=SsN(i,j); %indeks S
            R=RsN(i,j); %indeks R
            
            %[A,B]=funckja_ksztaltu(signal,Q,S,R);
            %[C,D]=funckja_ksztaltu2(signal,Q,S,R);
               
           
             %klasa(j,:)=[A,B,C,D,i,R,class] %wpisujmy tez ktory sygnal i ktory
             %to zalamek na zas i nazwe klasy na zas do kazdego tez
             
             [ampR]=R_peak_amplitude(signal,Q,S);
             [QRSenergy]=QRS_energy(signal,Q,S);
             [PtN]=Positive_to_Negative(signal,Q,S);
             [AtP]=Area_to_Perimeter(signal,Q,S);
             vec=[ampR,QRSenergy,PtN,AtP,i,R,class];
             Nclass(k,:)=vec;
             k=k+1;
        end
        
        
        % i tutaj tworzymy caly wektor dla jednej klasy, kolejne wiersze to
        % kolejne QRS w roznych sygnalach
        
    end
    
end

%% VE = 2 (id)
N=n;
class=2;
k=1;
for i=1:N %iteracja po kole
    %m=i;
    signal=signals{1,i};
    
    for j=1:length(QsVE)
        if QsVE(i,j)>0
           
            %i to kolejny sygnal w strukturze i kolejny wiersz w macierzach
            %j to kolejne probki czy kolejne za?amki QRS
            
            Q=QsVE(i,j); %indeks Q
            S=SsVE(i,j); %indeks S
            R=RsVE(i,j); %indeks R
            
            %[A,B]=funckja_ksztaltu(signal,Q,S,R);
            %[C,D]=funckja_ksztaltu2(signal,Q,S,R);
               
           
             %klasa(j,:)=[A,B,C,D,i,R,class] %wpisujmy tez ktory sygnal i ktory
             %to zalamek na zas i nazwe klasy na zas do kazdego tez
             
             [ampR]=R_peak_amplitude(signal,Q,S);
             [QRSenergy]=QRS_energy(signal,Q,S);
             [PtN]=Positive_to_Negative(signal,Q,S);
             [AtP]=Area_to_Perimeter(signal,Q,S);
             vec=[ampR,QRSenergy,PtN,AtP,i,R,class];
             VEclass(k,:)=vec;
             k=k+1;
        end
        
        
        % i tutaj tworzymy caly wektor dla jednej klasy, kolejne wiersze to
        % kolejne QRS w roznych sygnalach
        
    end
    
end

%% SV - 3

N=n;
class=3;
k=1;
for i=1:N %iteracja po kole
    %m=i;
    signal=signals{1,i};
    
    for j=1:length(QsSV)
        if QsSV(i,j)>0
           
            %i to kolejny sygnal w strukturze i kolejny wiersz w macierzach
            %j to kolejne probki czy kolejne za?amki QRS
            
            Q=QsSV(i,j); %indeks Q
            S=SsSV(i,j); %indeks S
            R=RsSV(i,j); %indeks R
            
            %[A,B]=funckja_ksztaltu(signal,Q,S,R);
            %[C,D]=funckja_ksztaltu2(signal,Q,S,R);
               
           
             %klasa(j,:)=[A,B,C,D,i,R,class] %wpisujmy tez ktory sygnal i ktory
             %to zalamek na zas i nazwe klasy na zas do kazdego tez
             
             [ampR]=R_peak_amplitude(signal,Q,S);
             [QRSenergy]=QRS_energy(signal,Q,S);
             [PtN]=Positive_to_Negative(signal,Q,S);
             [AtP]=Area_to_Perimeter(signal,Q,S);
             vec=[ampR,QRSenergy,PtN,AtP,i,R,class];
             SVclass(k,:)=vec;
             k=k+1;
        end
        
        
        % i tutaj tworzymy caly wektor dla jednej klasy, kolejne wiersze to
        % kolejne QRS w roznych sygnalach
        
    end
    
end
