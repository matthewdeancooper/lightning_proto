# Copyright (C) 2020 Matthew Cooper

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import deque

import matplotlib

# Need to use thread safe alternative
matplotlib.use("Agg")

import threading

from pynetdicom import (AE, ALL_TRANSFER_SYNTAXES,
                        AllStoragePresentationContexts, evt)

import dicom_network_model_export_scu as scu
import inference
import standard_utils

lock = threading.Lock()
import argparse


def handle_accepted(event):
    with lock:
        dicom_store[event.assoc] = []


def handle_store(event, storage_path):
    """Handle EVT_C_STORE events."""

    with lock:
        ds = event.dataset
        ds.file_meta = event.file_meta

        # Path for the study
        study_path = storage_path + "/" + ds.StudyInstanceUID
        standard_utils.make_directory(study_path)

        # Add study path to storage dictionary
        dicom_store[event.assoc].append(study_path)

        # Path for an imaging instance
        save_path = study_path + "/" + ds.SOPInstanceUID + ".dcm"
        ds.save_as(save_path, write_like_original=False)

    return 0x0000


def handle_release(event):
    """Handle EVT_RELEASE ."""
    with lock:
        # Sanity check
        assert len(set(dicom_store[event.assoc])) == 1

        # Add study path to queue for inference
        inference_queue.append(dicom_store[event.assoc][0])

        # Prevent memory leakage
        dicom_store.pop(event.assoc)

    # Show updated queue
    print_inference_queue()

    return 0x0000


def inference_loop(checkpoint_path, root_uid, scu_ip, scu_port, export):
    while True:
        time.sleep(1)

        if inference_queue:
            study_path = inference_queue.popleft()
            dicom_structure_file = inference.infer_contours(
                study_path,
                root_uid,
                checkpoint_path,
            )

            save_path = (study_path + "/" + dicom_structure_file.SOPInstanceUID +
                         "_model.dcm")

            # Local save of the structure file with imaging series
            dicom_structure_file.save_as(save_path, write_like_original=False)

            if export is not None:
                # # Return the imaging series too?
                if export == "series":
                    scu.export_files(study_path, scu_ip, scu_port, directory=True)
                # Return the structure file only
                else:
                    scu.export_files([save_path],
                                    scu_ip,
                                    scu_port,
                                    directory=False)

            print("\n--------------------------")
            print("INFERENCE COMPLETED:")
            print(study_path)

            print_inference_queue()
            if not inference_queue:
                print_listening()


def print_inference_queue():
    print("\n--------------------------")
    print("INFERENCE QUEUE:", len(inference_queue))
    for index, path in enumerate(inference_queue):
        print("Position", index, "-", path)


def print_listening():
    print("\n==========================")
    print("Listening for association requests...")


def main(storage_path, checkpoint_path, scp_ip, scp_port, scu_ip, scu_port,
         root_uid, export):

    # For testing
    # checkpoint_path = "/home/matthew/lightning_proto/lightning_proto/lightning_logs/version_1/checkpoints/epoch=0-step=128.ckpt"

    # Parent folder to all storage requests
    standard_utils.make_directory(storage_path)

    ae = AE()
    ae.network_timeout = None
    ae.acse_timeout = None
    ae.dimse_timeout = None
    ae.maximum_pdu_size = 0
    ae.maximum_associations = 12  # Tested with 12 threads

    handlers = [
        (evt.EVT_ACCEPTED, handle_accepted),
        (evt.EVT_C_STORE, handle_store, [storage_path]),
        (evt.EVT_RELEASED, handle_release),
    ]

    storage_sop_classes = [
        cx.abstract_syntax for cx in AllStoragePresentationContexts
    ]

    for uid in storage_sop_classes:
        ae.add_supported_context(uid, ALL_TRANSFER_SYNTAXES)

    ae.start_server((scp_ip, scp_port), block=False, evt_handlers=handlers)

    print_listening()

    inference_loop(checkpoint_path, root_uid, scu_ip, scu_port, export)


if __name__ == "__main__":
    global inference_queue
    inference_queue = deque()

    global dicom_store
    dicom_store = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path",
                        type=str,
                        default="dicom_storage_requests")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="../test_model/checkpoint.ckpt")
    parser.add_argument("--scp_ip", type=str, default="127.0.0.1")
    parser.add_argument("--scp_port", type=int, default=11112)
    parser.add_argument("--scu_ip", type=str, default="127.0.0.1")
    parser.add_argument("--scu_port", type=int, default=11112)
    parser.add_argument("--root_uid",
                        type=str,
                        default="1.2.826.0.1.3680043.8.498.")
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    main(**vars(args))
