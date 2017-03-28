\version "2.10.33"

\header {
  title = "303 - Herr Christ, der ein'ge Gott'ssohn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f8 g8 a4 g4 f4 
    e2 d4 a'4 
    bes4 g4 a4 g4 
    f2. } f8 g8 
  a4 a4 bes4 bes4   %5
  c2 g4 c4 
  a4 a4 g4 f4 
  e2 d4 a'4 
  bes4 g4 a4 g4 
  f2.  %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    d8 e8 f4 e4 d4 
    d4 cis4 a4 fis'4 
    g8 f8 e4 f4 e4 
    c2. } c4 
  f4 fis4 g4 g4   %5
  g4 f4 e4 g4 
  f4 ees4 d4 d4 
  d4 cis4 d4 fis4 
  g8 f8 e4 f4 e4 
  c2.  %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    bes4 c4 c8 bes8 a4 
    bes4 a8 g8 f4 d'4 
    d4 c4 c4 c8 bes8 
    a2. } a4 
  d4 d4 d4 d4   %5
  c2 c4 c4 
  c4 c4 bes4 b4 
  e,8 f8 g4 f4 c'4 
  d4 c4 c4 c8 bes8 
  a2.  %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 f4 
    bes,4 f'4 c4 d4 
    g,4 a4 d4 d4 
    g8 a8 bes4 a8 f8 c4 
    f,2. } f'8 e8 
  d4 c4 bes8 a8 g4   %5
  a4. bes8 c4 e4 
  f4 fis4 g4 gis4 
  a4 a,4 bes4 a4 
  g8 a8 bes4 a8 bes8 c4 
  f,2.  %10
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}