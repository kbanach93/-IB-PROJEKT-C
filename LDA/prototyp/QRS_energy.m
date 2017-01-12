function [ QRS_energy ] = QRS_energy( signal, QRS_start, QRS_end )

% funkcja oblicza energie zespolu QRS

QRS_energy = sum((signal(QRS_start:QRS_end)).^2);


end

