\version "2.10.33"

\header {
  title = "204 - Wer weiß, wie nahe mir"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 a4 bes4 a4 
    g4 a4 fis2 
    d4 f4 f4 ees4 
    d4 g4 g4 fis4 
    g2. } a4   %5
  bes4 c4 d4 d4 
  c4 c4 bes4 d4 
  c4 bes4 a4 g8 a8 
  bes4 a4 g2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    d4 d4 d4 d4 
    bes4 ees4 d2 
    a4 d4 d4 c4 
    bes4 bes8 c8 d4 d4 
    d2. } fis4   %5
  g4 f4 f4 g4 
  g4 f4 f4 f4 
  fis4 g4 g8 fis8 g4 
  g4 fis4 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    bes4 a4 g4 fis4 
    g4 c4 a2 
    fis4 a4 bes4 f4 
    f4 g8 a8 bes4 a4 
    bes2. } d4   %5
  d4 c4 bes4 bes4 
  bes4 a4 d4 bes4 
  c4 d4 ees8 d16 c16 bes8 c8 
  d4 d8 c8 b2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'4 fis4 g4 d4 
    ees4 c4 d2 
    d4 d4 g,4 a4 
    bes4 ees4 d4 d4 
    g,2. } d'4   %5
  g4 a4 bes4 g4 
  ees4 f4 bes,4 bes'4 
  a4 g4 c,8 d8 ees4 
  d8 c8 d4 g,2 
  
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