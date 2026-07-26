{{- define "loki.fullname" -}}
{{- printf "%s-loki" .Release.Name -}}
{{- end -}}
