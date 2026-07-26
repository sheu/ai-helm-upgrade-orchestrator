{{- define "kafka-connect.fullname" -}}
{{- printf "%s-kafka-connect" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
