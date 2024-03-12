from flask import jsonify
def packMassage(code,message,data):
    return jsonify({
      "code": code,
      "message": message,
      "data": data
    })