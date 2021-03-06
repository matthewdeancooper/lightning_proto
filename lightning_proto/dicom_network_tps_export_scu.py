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

import glob

from pynetdicom import AE, debug_logger
from pynetdicom.sop_class import CTImageStorage

import dicom_utils

debug_logger()


def export_files(data_path, scp_ip, scp_port):
    dicom_paths = glob.glob(data_path + "/*.dcm")
    print("dicom_paths", len(dicom_paths))

    dicom_files = dicom_utils.read_dicom_paths(dicom_paths, force=True)
    dicom_files = dicom_utils.add_transfer_syntax(dicom_files)
    dicom_series, *rest = dicom_utils.filter_dicom_files(dicom_files)
    print("dicom_series", len(dicom_series))

    # Initialise the Application Entity
    ae = AE()

    # Add a requested presentation context
    ae.add_requested_context(CTImageStorage)

    # Associate with peer AE at IP 127.0.0.1 and port 11112
    assoc = ae.associate(scp_ip, scp_port)
    if assoc.is_established:
        # Use the C-STORE service to send the dataset
        # returns the response status as a pydicom Dataset
        for ds in dicom_series:
            status = assoc.send_c_store(ds)

            # Check the status of the storage request
            if status:
                # If the storage request succeeded this will be 0x0000
                print("C-STORE request status: 0x{0:04x}".format(
                    status.Status))
            else:
                print(
                    "Connection timed out, was aborted or received invalid response"
                )

        # Release the association
        assoc.release()
    else:
        print("Association rejected, aborted or never connected")


if __name__ == "__main__":
    data_path = "../test_dicom_dataset/"
    scp_ip = "127.0.0.1"
    scp_port = 11112
    export_files(data_path, scp_ip, scp_port)
