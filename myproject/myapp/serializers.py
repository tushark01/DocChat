from rest_framework import serializers

class UserQuerySerializer(serializers.Serializer):
    url = serializers.URLField(required=True)
    query = serializers.CharField(required=True)