load Data_DNS
maturities = maturities(:);  % ensure a column vector
yields = Data(1:end,:);      % in-sample yields for estimation
lambda0 = 0.01;
X = [ones(size(maturities)) (1-exp(-lambda0*maturities))./(lambda0*maturities) ...
    ((1-exp(-lambda0*maturities))./(lambda0*maturities)-exp(-lambda0*maturities))];

beta = zeros(size(yields,1),3);
residuals = zeros(size(yields,1),numel(maturities));

for i = 1:size(yields,1)
    EstMdlOLS = fitlm(X, yields(i,:)', 'Intercept', false);
    beta(i,:) = EstMdlOLS.Coefficients.Estimate';
    residuals(i,:) = EstMdlOLS.Residuals.Raw';
end
EstMdlVAR = estimate(varm(3,1), beta);
Mdl = ssm(@(params)Example_DieboldLi(params,yields,maturities));
A0 = EstMdlVAR.AR{1};      % Get the VAR(1) matrix (stored as a cell array)
A0 = A0(:);                % Stack it columnwise
Q0 = EstMdlVAR.Covariance; % Get the VAR(1) estimated innovations covariance matrix
B0 = [sqrt(Q0(1,1)); 0; 0; sqrt(Q0(2,2)); 0; sqrt(Q0(3,3))];
H0 = cov(residuals);       % Sample covariance matrix of VAR(1) residuals
D0 = sqrt(diag(H0));       % Diagonalize the D matrix
mu0 = mean(beta)';
param0 = [A0; B0; D0; mu0; lambda0];
options = optimoptions('fminunc','MaxFunEvals',25000,'algorithm','quasi-newton', ...
    'TolFun' ,1e-8,'TolX',1e-8,'MaxIter',1000,'Display','off');

[EstMdlSSM,params] = estimate(Mdl,yields,param0,'Display','off', ...
    'options',options,'Univariate',true);

lambda = params(end);        % Get the estimated decay rate    
mu = params(end-3:end-1)';   % Get the estimated factor means
dispvars = {"SSM State Transition Matrix (A):";...
    "--------------------------------";...
    EstMdlSSM.A};
cellfun(@disp,dispvars)
dispvars = {"Two-Step State Transition Matrix (A):";...
    "--------------------------------";...
    EstMdlVAR.AR{1}};
cellfun(@disp,dispvars)
dispvars = {"SSM State Disturbance Loading Matrix (B):";...
    "-----------------------------------------";...
    EstMdlSSM.B};
cellfun(@disp,dispvars)
dispvars = {"SSM State Disturbance Covariance Matrix (Q = BB''):";...
    "--------------------------------------------------";...
    EstMdlSSM.B * EstMdlSSM.B'};
cellfun(@disp,dispvars)
dispvars = {"Two-Step State Disturbance Covariance Matrix (Q):";...
    "-------------------------------------------------";...
    EstMdlVAR.Covariance};
cellfun(@disp,dispvars)
dispvars = {"SSM Factor Means:";...
    "-----------------";...
    mu};
cellfun(@disp,dispvars)
dispvars = {"Two-Step Factor Means:";...
    "----------------------";...
    mu0'};
cellfun(@disp,dispvars)
intercept = EstMdlSSM.C * mu';
deflatedYields = bsxfun(@minus,yields,intercept');
deflatedStates = smooth(EstMdlSSM,deflatedYields);
estimatedStates = bsxfun(@plus,deflatedStates,mu);
figure
plot(Dates, [beta(:,1) estimatedStates(:,1)])
title('Level (Long-Term Factor)')
ylabel('Percent')
datetick x
legend({'Two-Step','State-Space Model'},'location','best')
figure
plot(Dates, [beta(:,2) estimatedStates(:,2)])
title('Slope (Short-Term Factor)')
ylabel('Percent')
datetick x
legend({'Two-Step','State-Space Model'},'location','best')
figure
plot(Dates, [beta(:,3) estimatedStates(:,3)])
title('Curvature (Medium-Term Factor)')
ylabel('Percent')
datetick x
legend({'Two-Step','State-Space Model'},'location','best')
dispvars = {"SSM Decay Rate (Lambda):";...
    "------------------------";...
    lambda};
cellfun(@disp,dispvars)
tau = 0:(1/12):max(maturities);    % Maturity (months)
decay = [lambda0 lambda];
loading = zeros(numel(tau), 2);

for i = 1:numel(tau)
    loading(i,:) = ((1-exp(-decay*tau(i)))./(decay*tau(i))-exp(-decay*tau(i))); 
end

figure
plot(tau,loading)
title('Loading on Curvature (Medium-Term Factor)')
xlabel('Maturity (Months)')
ylabel('Curvature Loading')
legend({'\lambda = 0.01 Fixed by Two-Step', ['\lambda = ' num2str(lambda) ' Estimated by SSM'],},'location','best')
residualsSSM = yields - estimatedStates*EstMdlSSM.C';
residuals2Step = yields - beta*X';

residualMeanSSM = 100*mean(residualsSSM)';
residualStdSSM = 100*std(residualsSSM)';
residualMean2Step = 100*mean(residuals2Step)';
residualStd2Step = 100*std(residuals2Step)';

dispvars = {"  -------------------------------------------------";...
    "              State-Space Model       Two-Step'";...
    "            -------------------  ------------------";...
    "                       Standard            Standard";...
    "  Maturity    Mean    Deviation    Mean   Deviation";...
    "  (Months)   (bps)      (bps)     (bps)     (bps)  ";...
    "  --------  --------  ---------  -------  ---------";...
    [maturities residualMeanSSM residualStdSSM residualMean2Step residualStd2Step]};
cellfun(@disp,dispvars)
horizon = 12;   % forecast horizon (months)

[forecastedDeflatedYields,mse] = forecast(EstMdlSSM,horizon,deflatedYields);
forecastedYields = bsxfun(@plus,forecastedDeflatedYields,intercept');
[~,~,results] = smooth(EstMdlSSM,deflatedYields);  % FILTER could also be used
EstMdlSSM.Mean0 = results(end).SmoothedStates;     % initialize the mean
cov0 = results(end).SmoothedStatesCov;
EstMdlSSM.Cov0 = (cov0 + cov0')/2;                 % initialize the covariance
rng('default')
nPaths = 100000;
simulatedDeflatedYields = simulate(EstMdlSSM, horizon, nPaths);
simulatedYields = bsxfun(@plus, simulatedDeflatedYields, intercept');
simulatedYields = permute(simulatedYields,[3 1 2]); % re-order for convenience
forecasts = zeros(horizon,numel(maturities));
standardErrors = zeros(horizon,numel(maturities));

for i = 1:numel(maturities)
    forecasts(:,i) = mean(simulatedYields(:,:,i));
    standardErrors(:,i) = std(simulatedYields(:,:,i));
end
figure
plot(maturities, [forecasts(horizon,:)' forecastedYields(horizon,:)'])
title('12-Month-Ahead Forecasts: Monte Carlo vs. MMSE')
xlabel('Maturity (Months)')
ylabel('Percent')
legend({'Monte Carlo','Minimum Mean Square Error'},'location','best')
figure
plot(maturities, [standardErrors(horizon,:)' sqrt(mse(horizon,:))'])
title('12-Month-Ahead Forecast Standard Errors: Monte Carlo vs. MMSE')
xlabel('Maturity (Months)')
ylabel('Percent')
legend({'Monte Carlo','Minimum Mean Square Error'},'location','best')
index12 = find(maturities == 12);  % page index of 12-month yield
bins = 0:0.2:12;

figure
subplot(3,1,1)
histogram(simulatedYields(:,1,index12), bins, 'Normalization', 'pdf')
title('Probability Density Function of 12-Month Yield')
xlabel('Yield 1 Month into the Future (%)')

subplot(3,1,2)
histogram(simulatedYields(:,6,index12), bins, 'Normalization', 'pdf')
xlabel('Yield 6 Months into the Future (%)')
ylabel('Probability')

subplot(3,1,3)
histogram(simulatedYields(:,12,index12), bins, 'Normalization', 'pdf')
xlabel('Yield 12 Months into the Future (%)')
