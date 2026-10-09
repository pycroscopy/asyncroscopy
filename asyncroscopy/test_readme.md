 

 

## Current Capabilities 
 
### Discovery and Readiness 
 
1. List available microscopes. 
2. List available devices. 
3. List available detectors. 
4. Which microscope should be used for a STEM workflow? 
5. Which detectors are exposed for the current instrument? 
6. Which supporting devices must be online before acquisition can start? 
7. Which commands should be checked before attempting image acquisition? 
8. What is the safest first step before running any acquisition workflow? 
9. How would you confirm the active instrument is reachable? 
10. Which device classes are currently exported in the stack? 
 
### STEM and HAADF Imaging 
 
11. Set the microscope to STEM mode and acquire a HAADF image. 
12. Acquire a low-magnification HAADF overview image. 
13. Acquire an atomic-resolution HAADF image. 
14. Reacquire the HAADF image after changing the field of view. 
15. Reacquire the HAADF image after changing focus. 
16. Reacquire the HAADF image after correcting astigmatism. 
17. Align the gun lens by adjusting the screen value to maximum. 
18. Center the beam before acquiring the next HAADF image. 
19. Optimize the HAADF image for contrast before saving it. 
20. Compare two HAADF images and report the likely cause of the change. 
 
### Alignment and Tuning 
 
21. How would you decide whether the image needs focus or astigmatism correction first? 
22. Which parameter would you tune first when the HAADF image looks blurred? 
23. Which parameter would you tune first when the image is asymmetric? 
24. How would you tell whether a bad image is caused by mode, focus, or beam placement? 
25. How would you adjust the microscope if the beam is off-axis? 
26. How would you handle a request to maximize signal without changing the sample position? 
27. What would you do if a requested acquisition seems inconsistent with the current microscope state? 
28. What should the agent ask for before making a destructive or irreversible change? 
29. How would you report that tuning improved the image but did not fully solve the issue? 
30. Which commands are useful for reading back current tuning values before changing them? 
 
### EDS Grid Acquisition 
 
31. Acquire EDS spectra on a 9×9 grid centered on the HAADF image at 300 pA beam current. 
32. Recenter the EDS map on the HAADF image and acquire again. 
33. Acquire a coarse EDS grid and explain why it is coarser than the previous one. 
34. Acquire a denser EDS grid and explain the tradeoff. 
35. Which detector or mode should be used for an EDS workflow? 
36. What beam current would you prefer before collecting a point spectrum versus a map? 
37. How would you confirm the EDS acquisition is aligned to the image center? 
38. How would you summarize an EDS grid acquisition in one sentence? 
39. How would you decide whether to repeat an EDS map after a failed acquisition? 
40. How would you report the relationship between image contrast and EDS sampling density? 
 
### Device and State Reasoning 
 
41. Report the current microscope state. 
42. Which state is the microscope in right now, and is it ready for acquisition? 
43. Which detector is active for the current acquisition path? 
44. Which supporting device values should be checked before an image is acquired? 
45. What current scan settings matter most for a HAADF image? 
46. What current detector settings matter most for an EDS workflow? 
47. Which state values should be considered before moving from discovery to acquisition? 
48. How would you explain the difference between a ready device and a ready microscope? 
49. How would you detect that a microscope is configured for the wrong mode? 
50. How would you recover from a state mismatch without assuming the hardware is wrong? 
 
### Data and Return Values 
 
51. How would you retrieve the data produced by a microscope acquisition? 
52. How would you identify the saved key for an acquired image? 
53. How would you describe the metadata attached to an acquired dataset? 
54. How would you determine whether an acquisition returned image data or a spectrum? 
55. How would you inspect a dataset preview before downloading the full result? 
56. How would you report the acquisition type stored in the returned file? 
57. How would you explain where the acquired data was written? 
58. How would you distinguish between an image acquisition and a detector configuration query? 
59. How would you confirm the returned key is valid before using it downstream? 
60. How would you summarize the outputs of an acquisition for a user who only wants the result type and location? 
 
### Safety and Sequencing 
 
61. Which step should come first: discovery, state check, or acquisition? 
62. Which step should come first: beam placement or detector selection? 
63. Which step should come first: focus correction or EDS mapping? 
64. Which step should come first: mode selection or image acquisition? 
65. Which step should come first: checking detector readiness or saving the result? 
66. How would you avoid repeating an acquisition with stale settings? 
67. How would you decide whether a follow-up acquisition should reuse the current beam current? 
68. How would you respond if a user asks for two conflicting actions in a single turn? 
69. How would you separate safe planning from an action that changes microscope state? 
70. How would you explain why an acquisition was deferred until the instrument was ready? 
 
## Near-Term Goals 
 
### Image Acquisition with Detector Choice 
 
71. Acquire an image after checking the microscope mode first. 
72. Acquire an image after deciding which detector should be used. 
73. List available image detectors. 
74. Choose the detector for a STEM image when multiple options are available. 
75. Explain why one detector is better than another for the requested image. 
76. Acquire a STEM image with a detector chosen from the available image detectors. 
77. Recompute the detector choice after the microscope mode changes. 
78. How would you answer if the microscope mode is unknown but an image is requested? 
79. How would you decide whether the requested image belongs in STEM or another mode? 
80. How would you report the detector choice before running the acquisition? 
 
### EELS Grid Acquisition 
 
81. Acquire EELS on a 9×9 grid at 10 pA beam current. 
82. Choose the EELS grid center from the current image context. 
83. Explain why EELS might require a different beam current than EDS. 
84. Describe the sequencing needed before collecting EELS on a grid. 
85. What detector or spectrometer choice would you expect for EELS? 
86. How would you compare the EELS plan to the EDS plan already supported today? 
87. How would you report that EELS is a near-term target rather than a current capability? 
88. Which microscope state details should be rechecked before EELS begins? 
89. How would you handle a request for EELS if the required device is not yet exposed? 
90. How would you make the EELS request fail gracefully while preserving the rest of the workflow? 
 
### Convergence and CBED Preparation 
 
91. Set the convergence angle to 10 mrad. 
92. Acquire a convergent beam electron diffraction pattern with the Ceta camera. 
93. Explain the detector choice for a CBED acquisition. 
94. Explain how the convergence angle affects the CBED pattern. 
95. How would you prepare the microscope for a diffraction-style acquisition? 
96. What additional alignment would you want before CBED if the probe is unstable? 
97. How would you report the expected output of a CBED acquisition? 
98. How would you distinguish CBED from a standard STEM image request? 
99. Which microscope settings would you want to confirm before CBED begins? 
100. How would you reject a CBED request if the camera path is unavailable? 
 
### Screen and Beam Control 
 
101. Ensure the main screen is down whenever possible. 
102. Explain why the main screen should stay down whenever possible. 
103. Report whether the main screen is currently up or down. 
104. Lower the screen before the next acquisition if it is safe to do so. 
105. Keep the screen down during image acquisition unless the workflow requires otherwise. 
106. Explain whether the screen state changes the valid detector choice. 
107. Explain whether the screen state should be checked before a diffraction acquisition. 
108. Explain whether the screen state should be checked before a spectroscopy acquisition. 
109. What is the safest order for checking screen state, detector state, and microscope mode? 
110. How would you explain a refusal to change the screen when the state is already correct? 
 
### Microscope State Reporting 
 
111. Report current microscope state. 
112. Report current microscope mode and readiness in one sentence. 
113. Report the most relevant current values before a requested acquisition. 
114. Report whether the microscope is ready for imaging, spectroscopy, or diffraction. 
115. Report the active detector family and why it matters. 
116. Report the current state without assuming the microscope is in STEM mode. 
117. Report what must change before the requested workflow can proceed. 
118. Report the smallest set of state facts needed to justify the next action. 
119. Report whether the next action is blocked by configuration or by missing hardware. 
120. Report the state in a way that helps the user decide the next experiment step. 
 
### State-Aware Planning 
 
121. How would you plan an acquisition if the mode, detector, and current are all specified? 
122. How would you plan an acquisition if only the experiment goal is specified? 
123. How would you choose between image, spectrum, and diffraction acquisition from the current state? 
124. How would you decide whether to reuse a previously centered region of interest? 
125. How would you sequence the steps for an image that precedes an EELS map? 
126. How would you sequence the steps for a CBED pattern that follows a beam alignment check? 
127. How would you explain when a requested workflow is blocked by unavailable support devices? 
128. How would you adapt the plan if the microscope state changed after the first check? 
129. How would you handle a request that needs a detector not currently exported? 
130. How would you ask for the minimum extra information needed to continue safely? 
 
### Graceful Fallbacks 
 
131. If the detector list is incomplete, how should the agent respond? 
132. If the microscope mode cannot be confirmed, how should the agent respond? 
133. If the acquisition type is near-term but not yet exposed, how should the agent respond? 
134. If the requested beam current is outside the supported range, how should the agent respond? 
135. If the requested camera is unavailable, how should the agent respond? 
136. If the screen cannot be lowered automatically, how should the agent respond? 
137. If a request is valid but cannot be executed safely right now, how should the agent respond? 
138. If the agent needs to stop after discovery, how should it explain the limitation? 
139. If a user asks for EELS and CBED in the same turn, how should the agent prioritize? 
140. If a request is partially supported, how should the agent separate supported from unsupported steps? 
 
## Future Functionality 
 
### Zone-Axis Tilt 
 
141. Tilt to zone axis. 
142. Tilt the specimen to a requested zone axis and report success. 
143. Explain what information you would need before attempting a zone-axis tilt. 
144. Describe how zone-axis tilt would change the downstream imaging plan. 
145. Explain how you would verify that the tilt reached the intended orientation. 
146. Explain how you would recover if the first tilt estimate overshoots the zone axis. 
147. Explain how you would use a pre-tilt image to guide the zone-axis search. 
148. Explain what state checks should precede a zone-axis tilt. 
149. Explain what safety checks should precede a zone-axis tilt. 
150. Explain how a zone-axis tilt request should fail if the capability is not yet available. 
 
### General Tilt Control 
 
151. Perform general tilt control. 
152. Tilt by a small increment and report the new position. 
153. Tilt to a target alpha value and then hold position. 
154. Tilt to a target beta value and then hold position. 
155. Describe how general tilt would be different from zone-axis tilt. 
156. Describe how you would combine tilt control with image reacquisition. 
157. Describe how you would combine tilt control with diffraction acquisition. 
158. Describe how you would combine tilt control with EDS or EELS mapping. 
159. Explain how you would check tilt limits before commanding motion. 
160. Explain how you would recover from a tilt request that exceeds safe bounds. 
 
### Tilt Safety and Limits 
 
161. What tilt limits should be checked before motion begins? 
162. What sample or holder conditions should be confirmed before tilting? 
163. What image or diffraction cue would indicate that the tilt should stop early? 
164. What should happen if the requested tilt conflicts with the current microscope mode? 
165. What should happen if the requested tilt conflicts with the available stage hardware? 
166. How should the agent describe a tilt that is possible in principle but unsafe right now? 
167. How should the agent describe a tilt that is unsupported in the current build? 
168. How should the agent decide whether to ask for a target orientation or a target angle first? 
169. How should the agent verify that the sample is still centered after tilt? 
170. How should the agent report the risk of sample drift after tilt? 
 
### Multi-Step Future Workflows 
 
171. Tilt to zone axis, then acquire a HAADF image. 
172. Tilt to zone axis, then acquire a CBED pattern. 
173. Tilt to zone axis, then acquire EELS on a grid. 
174. Tilt to zone axis, then acquire EDS on a grid. 
175. Perform a tilt sequence and then re-center the beam. 
176. Perform a tilt sequence and then refine focus before acquisition. 
177. Perform a tilt sequence and then recheck detector choice. 
178. Perform a tilt sequence and then report the final microscope state. 
179. Explain which part of a multi-step tilt workflow should be validated first. 
180. Explain how a multi-step tilt workflow should fail if any intermediate step is unsupported. 
 
### Diffraction-Centric Workflows 
 
181. Acquire a diffraction pattern after tilt compensation. 
182. Acquire a zone-axis diffraction pattern after general tilt adjustment. 
183. Compare a CBED pattern before and after tilt. 
184. Explain how diffraction contrast would change after tilting the sample. 
185. Explain how a diffraction workflow should differ from a STEM imaging workflow. 
186. Explain how a future diffraction workflow should report orientation metadata. 
187. Explain how a future diffraction workflow should capture the final beam current and convergence angle. 
188. Explain how a future diffraction workflow should note whether the main screen was down. 
189. Explain how a future diffraction workflow should identify the active detector family. 
190. Explain how a future diffraction workflow should summarize the acquisition chain. 
 
### Closed-Loop Automation 
 
191. Optimize tilt, focus, and detector selection automatically for the requested target. 
192. Adjust microscope settings iteratively until the image quality meets a threshold. 
193. Decide when to stop optimization and hand control back to the user. 
194. Decide when a closed-loop acquisition should abort because the state is unstable. 
195. Decide how to rank competing goals such as resolution, speed, and beam dose. 
196. Decide how to report uncertainty when the best settings are only estimated. 
197. Decide how to preserve provenance across a future multi-step workflow. 
198. Decide how to reuse measurements from a prior image when planning the next step. 
199. Decide how to compare the outcome of a current run with a prior baseline run. 
200. Decide how to present a short rationale for each automatic microscope change. 
 
### Reporting and Missing Features 
 
201. Which future capability is the closest match for a user asking for atomic-column diffraction mapping? 
202. Which future capability is the closest match for a user asking for automated zone-axis search? 
203. Which future capability is the closest match for a user asking for general specimen tilt guidance? 
204. Which future capability is the closest match for a user asking for diffraction-aware autofocus? 
205. Which future capability is the closest match for a user asking for a tilt-aware EELS map? 
206. How should the agent explain that a future capability is not yet available in the current build? 
207. How should the agent explain the gap between a benchmark question and a supported command? 
208. How should the agent preserve the rest of the workflow when one future step is unavailable? 
209. How should the agent phrase an unsupported-request answer without losing the scientific context? 
210. How should the agent suggest the next best supported action when a future workflow is requested today? 