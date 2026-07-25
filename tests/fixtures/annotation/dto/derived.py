# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: '/home/jjf190004/pyllm_dataset/dys-connector/670c67a8973ba2794b6ee0bd9732962e2f25cee9dfe74127f0dc38d8cfd0382c549615f1a616add875fd23a62b5fdd3e9df178110ff3937d326572f932e7186a/dto.py'
# Bytecode version: 3.15 (3666)
# Source timestamp: 2026-07-08 16:38:34 UTC (1783528714)

import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel
KISISEL_BELGE = '6a5547d5-09c1-4e18-816f-58fbaf1975bb'
class SharingRole(Enum):
    OWNER = 'OWNER'
    ADMIN = 'ADMIN'
    EDITOR = 'EDITOR'
    COLLEAGUE = 'COLLEAGUE'
    CONSUMER = 'CONSUMER'
class VerificationType(Enum):
    NONE = (0, 'verification.none')
    EMAIL_OTP = (1, 'verification.via.email')
    SMS_OTP = (3, 'verification.via.sms')
    IDM = (4, 'verification.via.logo.idm')
class VarValues(BaseModel):
    """\nAbstract VarValues extend this class to define your own VarValues.\n"""
class UploadDocumentDTO:
    def __init__(self, name, tag_ids=None, var_values: VarValues=VarValues()):
        if tag_ids is None:
            tag_ids = KISISEL_BELGE
        self.name = name
        self.tagIds = [tag_ids]
        self.varValues = var_values.dict()
        for key in self.varValues.copy():
            if not self.varValues[key]:
                del self.varValues[key]
    def get_as_json(self):
        return json.dumps(self.__dict__, ensure_ascii=False)
class SharingDTO(BaseModel):
    # ***<module>.SharingDTO: Failure: Different bytecode
    recipientId = ''
    role = SharingRole.CONSUMER.value
    customEnabled = True
    recipientId: Optional[str]