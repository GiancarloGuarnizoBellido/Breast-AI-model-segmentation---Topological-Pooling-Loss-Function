
def TopologicalPoolingLoss(y_hat,y, device='cuda'):


    def loss_per_batch(y_hat,y):

        axial_pool = nn.MaxPool3d(kernel_size=(1, 1, 128))
        sagittal_pool = nn.MaxPool3d(kernel_size=(128, 1, 1))
        coronal_pool = nn.MaxPool3d(kernel_size=(1, 128, 1))

        # Aplicar pooling
        P_axial = axial_pool(y_hat)    
        P_sagittal = sagittal_pool(y_hat)
        P_coronal = coronal_pool(y_hat) 

        P_axial=P_axial[:, :, :, 0]
        P_sagittal = P_sagittal[:, 0, :, :]
        P_coronal = P_coronal[:, :, 0, :]

        # Aplicar pooling
        G_axial = axial_pool(y)
        G_sagittal = sagittal_pool(y)
        G_coronal = coronal_pool(y) 

        G_axial = G_axial[:, :, :, 0]
        G_sagittal = G_sagittal[:, 0, :, :]
        G_coronal = G_coronal[:, :, 0, :]

        kernel_sizes=[4, 5, 8, 10, 20] #[2, 4, 8, 12, 15, 20]
        loss_values = []

        for k_size in kernel_sizes:
            pool_2d=nn.MaxPool2d(kernel_size=k_size)
            P_topo_k=(pool_2d(P_axial) + pool_2d(P_sagittal) + pool_2d(P_coronal))
            G_topo_k=(pool_2d(G_axial) + pool_2d(G_sagittal) + pool_2d(G_coronal))
            # Calcular la diferencia absoluta

            diff=torch.abs(((G_topo_k)-(P_topo_k))) #Obtengo cuantos pixeles no coinciden con GT en los 3 planos
            loss_k_mean = torch.mean(diff)  # Promedio de los errores para cada tamaño de k
            loss_values.append(loss_k_mean)
        # Promediar sobre todos los tamaños de kernels
        L_topo = torch.mean(torch.stack(loss_values))
        return L_topo

    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)

    y_hat = torch.sigmoid(y_hat) 
    # Init dice score for batch (GPU ready)
    if y_hat.is_cuda: l_topo_val = torch.FloatTensor(1).cuda(device='cuda').zero_()
    else: l_topo_val = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(y_hat, y)):
        l_topo_val +=  loss_per_batch(inputs[0], inputs[1])

    # Return the mean Dice coefficient over the given batch
    l_topo_batch = l_topo_val / (pair_idx + 1)

    #_, dice_loss = utils.dice_coeff_batch(y_hat, y)
    #lambda_val=1
    #L_topo=l_topo_batch+lambda_val*(dice_loss)
    
    #weight = 0.5
    #L_topo = l_topo_batch*weight + dice_loss*(1-weight)

    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    #weight = 0.33
    weight_1=0.4
    weight_2=0.3
    weight_3=0.3
    L_topo = l_topo_batch*weight_1 + bce_loss*weight_2 + dice_loss*weight_3

    return L_topo
