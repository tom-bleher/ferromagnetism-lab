function [chi2,Msol] = magnetfit(I,Ierr,T,Tc,H,Isat)

% Please notice this function returns 2 outputs! MATLAB requires a vector [a,b] to save them ( [a,b]=magnetfit(I,Ierr,T,Tc,H,Isat) )
% Also notice, the theoretical output "Msol" is already normalized by the saturation value.
% Convince yourself that inputting current/voltage/field will yield the same result since we are always normalizing by saturation 
% This function computes the chi2 value for a fit of the normalized 
% magnetization curve
% Input:  I    = Measured induced current
%         Ierr = Error of the measured current
%         T    = Temperature of the material
%         Tc   = Curie's temparature
%         H    = External field
%         Isat = The saturation value of the current
% Output: chi2 = chi square value
%         Msol = The model of the magnatization curve


kb=1.38e-23;    % Boltzman's constant [Joul/Kelvin]
mu = 0.928e-23; % Magnetic moment [Joul/Tesla]

Msol = zeros(size(T));
for K=1:length(T);
    % re-write the magnetization equation
    mf = inline(['abs(m-tanh(' num2str(mu) '*' num2str(H) '/(' num2str(kb) '*' num2str(T(K)) ')+(' num2str(Tc) '/' num2str(T(K)) ')*m))']);
    Msol(K)=fminbnd(mf,0,1);  % find the zero of the function, 
                              % i.e., for a given Tc,H, and T what is m    
end


% compute chi2
chi2 = sum((I/Isat - Msol).^2 ./ (Ierr/Isat).^2);