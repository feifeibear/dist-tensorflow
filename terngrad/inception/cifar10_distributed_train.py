# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import inception_distributed_train
from inception.cifar10_data import Cifar10Data

FLAGS = tf.app.flags.FLAGS
import os
import re

def tf_config_from_slurm(ps_number, port_number=2222):
    """
    Creates configuration for a distributed tensorflow session 
    from environment variables  provided by the Slurm cluster
    management system.

    @param: ps_number number of parameter servers to run
    @param: port_number port number to be used for communication
    @return: a tuple containing cluster with fields cluster_spec,
             task_name and task_id 
    """

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    print(nodelist)
    print("jacob")
    nodename = os.environ["SLURMD_NODENAME"]
    print(nodename)

    nodelist = _expand_nodelist(nodelist)
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))

    if len(nodelist) != num_nodes:
        raise ValueError("Number of slurm nodes {} not equal to {}".format(len(nodelist), num_nodes))

    if nodename not in nodelist:
        raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(nodename,nodelist))
    if ps_number > num_nodes :
        raise ValueError("Number of ps node is largger than nodes be given by slurm!")
    ps_nodes = [node for i, node in enumerate(nodelist) if i < ps_number]
    worker_nodes = [node for i, node in enumerate(nodelist) if i >= ps_number]

    if nodename in ps_nodes:
        my_job_name = "ps"
        my_task_index = ps_nodes.index(nodename)
    else:
        my_job_name = "worker"
        my_task_index = worker_nodes.index(nodename)

    worker_sockets = [":".join([node, str(port_number)]) for node in worker_nodes]
    ps_sockets = [":".join([node, str(port_number)]) for node in ps_nodes]
    cluster = {"worker": worker_sockets, "ps" : ps_sockets}

    return cluster, my_job_name, my_task_index

def _pad_zeros(iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)
def _expand_ids(ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            begin, end = [int(token) for token in id.split('-')]
            myl = len((id.split('-'))[0])
            result.extend(_pad_zeros(range(begin, end+1), myl))
        else:
            result.append(id)
    return result

def _expand_nodelist(nodelist):
    prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
    ids = _expand_ids(ids)
    result = [prefix + str(id) for id in ids]
    return result

def _worker_task_id(nodelist, nodename):
    return nodelist.index(nodename)




def main(unused_args):
  FLAGS.dataset_name = 'cifar10'
#
#  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'
#
#  # Extract all the hostnames for the ps and worker jobs to construct the
#  # cluster spec.
#  ps_hosts = FLAGS.ps_hosts.split(',')
#  worker_hosts = FLAGS.worker_hosts.split(',')
#  tf.logging.info('PS hosts are: %s' % ps_hosts)
#  tf.logging.info('Worker hosts are: %s' % worker_hosts)
#
#  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
#                                       'worker': worker_hosts})
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
#
#  server = tf.train.Server(
#      {'ps': ps_hosts,
#       'worker': worker_hosts},
#      job_name=FLAGS.job_name,
#      task_index=FLAGS.task_id,
#      config=sess_config)

  cluster_spec, my_job_name, my_task_index = tf_config_from_slurm(ps_number=2)
  tf.logging.info("debug info")
  tf.logging.info(cluster_spec)
  tf.logging.info(my_job_name)
  tf.logging.info(my_task_index)

  server = tf.train.Server(server_or_cluster_def=cluster_spec,
                           job_name=my_job_name,
                           task_index=my_task_index,
                          config=sess_config)



  if my_job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = Cifar10Data(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if my_task_index == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec, my_task_index)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
