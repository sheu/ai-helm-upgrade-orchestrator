{{- define "grafana.fullname" -}}
{{- printf "%s-grafana" .Release.Name -}}
{{- end -}}
