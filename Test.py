import tensorflow as tf
import numpy as np
temp1 = tf.range(0,36)
temp2 = tf.constant(1,shape=[36])
temp3 = tf.reshape(temp1 + temp2, [3,3,4])
temp4 = tf.gather(temp3, [0,1], axis = 0)
temp5 = tf.gather(temp3, 1, axis=1)
temp6 = tf.gather(temp3, [0,3], axis=2)
temp7 = tf.gather(temp3, [0,1,2], axis=2)
ar = [[1, 2, 3, 4, 5, 6 ,7, 8],
      [7, 6 ,5 ,4 ,3 ,2, 1, 0],
      [3, 3, 3, 3, 3, 3, 3, 3],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [2, 2, 2, 2, 2, 2, 2, 2]]
temp8 = tf.constant(ar)
temp9 = tf.reshape(temp8, [5,2,4])
temp10 = tf.tile(temp8,[2,2])
temp11 = tf.sequence_mask([1, 2, 3], 5)
temp12 = tf.sequence_mask([[1, 2], [3, 4]])
temp13 = tf.where(temp8)
with tf.Session() as sess:
    print('temp1:', sess.run(temp1))
    print('temp2:', sess.run(temp2))
    print('temp1+temp2:', sess.run(temp1 + temp2))
    print('temp3:', sess.run(temp3))
    print('temp4:', sess.run(temp4))
    print('temp5:', sess.run(temp5))
    print('temp6:', sess.run(temp6))
    print('temp7:', sess.run(temp7))
    print('temp8:', sess.run(temp8))
    print('temp9:', sess.run(temp9))
    print('temp10:', sess.run(temp10))
    print('temp11:', sess.run(temp11))
    print('temp12:', sess.run(temp12))
    print('temp13:', sess.run(temp13))

t1 = tf.constant([[1, 2, 3], [4, 5, 6]])  #这是一个2*3的矩阵
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])  #2*3
with tf.Session() as sess:
    print(sess.run(tf.concat(values=[t1, t2], axis=0)))
    print(sess.run(tf.concat(values=[t1, t2], axis=1)))
    print(sess.run(tf.sequence_mask([1,2,9],3)))

def foo(*args,**kwargs):
    print('args:',args)
    print('kwargs:',kwargs)
    print('------------------')

if __name__ == '__main__':
    foo(1,2,3,4)
    foo(a=1,b=2,c=3,d=4)
    foo(1,2,3,4,a=1,b=2,c=3)
    foo('a',1,None,a=1,b='2',c=3)