{{- define "prometheus.fullname" -}}
{{- printf "%s-prometheus" .Release.Name -}}
{{- end -}}
