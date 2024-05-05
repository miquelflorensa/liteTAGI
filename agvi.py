import numpy as np
import matplotlib.pyplot as plt

def main():
    SW_true = 0.5
    mW2_hat_init = 7**2
    SW2_hat_init = 4**2

    num_samples = 500

    # Generate samples
    samples = np.random.normal(0, SW_true, num_samples)

    # Plot samples
    #plt.plot(samples)
    #plt.show()

    mu_W2_hat = []
    sigma2_W2_hat = []

    mu_W2_hat.append(mW2_hat_init)
    sigma2_W2_hat.append(SW2_hat_init)


    for i in range(num_samples):
        # \mu_{t|t}^{W} = y_t 
        # \sigma_{t|t}^{W} = 0
        mu_W_prior = samples[i]
        sigma2_W_prior = 0

        # \mu_{t|t}^{W^2} = (\mu_{t|t}^{W})^2 + (\sigma_{t|t}^{W})^2
        # (\sigma_{t|t}^{W^2})^2 = 2(\sigma_{t|t}^{W})^4 + 4(\sigma_{t|t}^{W})^2  (\mu_{t|t}^{W})^2
        mu_W2_prior = mu_W_prior**2 + sigma2_W_prior
        sigma2_W2_prior = 2*(sigma2_W_prior)**2 + 4*(sigma2_W_prior)*(mu_W_prior)**2

        # \mu_{t|t-1}^{W^2} = \mu_{t-1|t-1}^{\hat{W^2}}
        # (\sigma_{t|t-1}^{W^2})^2 = 3(\sigma_{t-1|t-1}^{\hat{W^2}})^2 + 2(\mu_{t-1|t-1}^{\hat{W^2}})^2
        mu_W2_pos = mu_W2_hat[-1]
        sigma2_W2_pos = 3*sigma2_W2_hat[-1] + 2*(mu_W2_hat[-1])**2

        # K_t = \frac{(\sigma_{t|t-1}^{W^2})^2}{(\sigma_{t-1|t-1}^{\hat{W^2}})^2}
        # \mu_{t|t}^{\hat{W^2}} = \mu_{t-1|t-1}^{\hat{W^2}} + K_t(\mu_{t|t}^{W^2} - \mu_{t|t-1}^{W^2})
        # (\sigma_{t|t}^{\hat{W^2}})^2 = (\sigma_{t|t-1}^{\hat{W^2}})^2 + 
        # (K_t)^2 ((\sigma_{t|t}^{W^2})^2 - (\sigma_{t|t-1}^{W^2})^2)
        K_t = sigma2_W2_hat[-1]/sigma2_W2_pos
        mu_W2_hat.append(mu_W2_hat[-1] + K_t*(mu_W2_prior - mu_W2_pos))
        sigma2_W2_hat.append(sigma2_W2_hat[-1] + K_t**2*(sigma2_W2_prior - sigma2_W2_pos))               

    mu_W2_hat = np.sqrt(np.array(mu_W2_hat))
    sigma2_W2_hat = np.sqrt(np.array(sigma2_W2_hat))

    # Plot the results
    plt.figure()
    n = np.arange(0,num_samples+1,1)
    plt.plot(n,np.ones([num_samples+1,1])*SW_true, '--', color = 'black',label='True $\sigma_W$' )
    plt.plot(mu_W2_hat,label=r'$\mu_{\bar{W^2}}$')
    plt.fill_between(n, mu_W2_hat-sigma2_W2_hat, mu_W2_hat+sigma2_W2_hat, facecolor='gray',  alpha=0.5,label=r'$\mu_{\bar{W^2}} \pm \sigma_\bar{W^2}$')
    plt.fill_between(n, mu_W2_hat-2*sigma2_W2_hat, mu_W2_hat+2*sigma2_W2_hat, facecolor='gray',  alpha=0.3,label=r'$\mu_{\bar{W^2}} \pm 2\sigma_\bar{W^2}$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
