vae_mnist = {
    'encoder':[
        {'name':'block', 'config':{'input_channels': 1, 'hidden_channels':32, 'act':'relu', 'norm':'batchnorm', 
                                   'residual':False, 'double_conv':True, 'pooling':'maxpooling'}},
        {'name':'block', 'config':{'input_channels': 32, 'hidden_channels':64, 'act':'relu', 'norm':'batchnorm', 
                                   'residual':False, 'double_conv':True, 'pooling':'maxpooling'}},
        {'name':'block', 'config':{'input_channels': 64, 'hidden_channels':128, 'act':'relu', 'norm':'batchnorm', 
                                   'residual':False, 'double_conv':True, 'pooling':'none'}},
        [
            {'name':'linear', 'config':{'in_features':128*7*7, 'out_features':128*7*7}},
            {'name':'linear', 'config':{'in_features':128*7*7, 'out_features':128*7*7}}
        ]
    ],
    'decoder':[
        {'name':'view', 'config':[-1,128,7,7]},
        {'name':'block', 'config':{'input_channels': 128, 'hidden_channels':64, 'conv_transpose':True, 
                                   'act':'relu', 'norm':'batchnorm', 'residual':False, 'double_conv':True}},
        {'name':'block', 'config':{'input_channels': 64, 'hidden_channels':32, 'conv_transpose':True, 
                                   'act':'relu', 'norm':'batchnorm', 'residual':False, 'double_conv':True}},
        {'name':'block', 'config':{'input_channels': 32, 'hidden_channels':1, 'conv_transpose':False, 
                                   'act':'relu', 'norm':'batchnorm', 'residual':False, 'double_conv':True}}
    ],
    'alpha': 0.5
}