function [ R_peak_ampl ] = R_peak_amplitude( signal, QRS_start, QRS_end )

% funkcja oblicza amplitude zalamka R

R_peak_ampl = max(signal(QRS_start:QRS_end));

% w razie czego opcja 2 wykorzystanie R wykrytego przez toolboxa



end

