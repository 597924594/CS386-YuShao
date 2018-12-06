def build_generator_new(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f1 = 7
        f2 = 3
        # o_c1 = layers.general_conv3d(inputgen, num of filters , size of filters , stride , stddev = 0.02, padding = "SAME", name = "c1")
        o_c1 = layers.general_conv3d(
            inputgen, 64, f1, 1, 0.02, "SAME", "c1")#C7S1-64
        o_c2 = layers.general_conv3d(
            o_c1, 128, f2, 2, 0.02, "SAME", "c2")#C3-128
        o_c3 = layers.general_conv3d(
            o_c2, 256, f2, 2, 0.02, "SAME", "c3")#C3-256
        ###################Residual Block#################################
        #Residual Block 1
        o_c4 = layers.general_conv3d(
            o_c3, 256, f2, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 2
        o_c6 = layers.general_conv3d(
            o_c5, 256, f2, 1, 0.02, "SAME", "c4")
        o_c7 = layers.general_conv3d(
            o_c6, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 3
        o_c8 = layers.general_conv3d(
            o_c7, 256, f2, 1, 0.02, "SAME", "c4")
        o_c9 = layers.general_conv3d(
            o_c8, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 4
        o_c10 = layers.general_conv3d(
            o_c9, 256, f2, 1, 0.02, "SAME", "c4")
        o_c11 = layers.general_conv3d(
            o_c10, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 5
        o_c12 = layers.general_conv3d(
            o_c11, 256, f2, 1, 0.02, "SAME", "c4")
        o_c13 = layers.general_conv3d(
            o_c12, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 6
        o_c14 = layers.general_conv3d(
            o_c13, 256, f2, 1, 0.02, "SAME", "c4")
        o_c15 = layers.general_conv3d(
            o_c14, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 7
        o_c16 = layers.general_conv3d(
            o_c15, 256, f2, 1, 0.02, "SAME", "c4")
        o_c17 = layers.general_conv3d(
            o_c16, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 8
        o_c18 = layers.general_conv3d(
            o_c17, 256, f2, 1, 0.02, "SAME", "c4")
        o_c19 = layers.general_conv3d(
            o_c18, 256, f2, 1, 0.02, "SAME", "c5")
        #Residual Block 9
        o_c20 = layers.general_conv3d(
            o_c19, 256, f2, 1, 0.02, "SAME", "c4")
        o_c21 = layers.general_conv3d(
            o_c20, 256, f2, 1, 0.02, "SAME", "c5")
        ###################Residual Block#################################
        o_c22 = layers.general_deconv3d(
            o_c21, 64, f2, 2, 0.02, "SAME", "c10")#TC64
        o_c23 = layers.general_deconv3d(
            o_c22, 32, f2, 2, 0.02, "SAME", "c10")#TC32
        out_gen = layers.general_deconv3d(
            o_c23, 3, f1, 1, 0.02, "SAME", "c10")#C7S1-3
        
        print ('*****')
        print (out_gen.get_shape())

        print (out_gen.name)
        return out_gen