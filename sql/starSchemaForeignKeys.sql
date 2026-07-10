ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_label FOREIGN KEY (label_id) REFERENCES dim_label(label_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_day FOREIGN KEY (day_id) REFERENCES dim_day(day_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_portgroup FOREIGN KEY (port_group_id) REFERENCES dim_port_group(port_group_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_protocol FOREIGN KEY (protocol_id) REFERENCES dim_protocol(protocol_id);
